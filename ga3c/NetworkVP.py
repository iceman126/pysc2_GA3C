# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf
import cv2

from Config import Config
from pysc2.lib import actions, features

class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_size = Config.IMAGE_SIZE
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.vl_coef = Config.VL_COEF

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)
                    )
                )
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    def _atari(self):
        screen_processed = self._embed_features(self.screen, features.SCREEN_FEATURES, "screen")
        minimap_processed = self._embed_features(self.minimap, features.MINIMAP_FEATURES, "minimap")
        screen_conv1 = tf.layers.conv2d(
                screen_processed,
                filters=16,
                kernel_size=8,
                strides=4,
                kernel_initializer=tf.glorot_normal_initializer(),
                # kernel_initializer=tf.orthogonal_initializer(),
                name="screen_conv1",
                padding="same",
                activation=tf.nn.relu
        )
        screen_conv2 = tf.layers.conv2d(
                screen_conv1,
                filters=32,
                kernel_size=4,
                strides=2,
                kernel_initializer=tf.glorot_normal_initializer(),
                # kernel_initializer=tf.orthogonal_initializer(),
                name="screen_conv2",
                padding="same",
                activation=tf.nn.relu
        )
        minimap_conv1 = tf.layers.conv2d(
                minimap_processed,
                filters=16,
                kernel_size=8,
                strides=4,
                kernel_initializer=tf.glorot_normal_initializer(),
                # kernel_initializer=tf.orthogonal_initializer(),
                name="minimap_conv1",
                padding="same",
                activation=tf.nn.relu
        )
        minimap_conv2 = tf.layers.conv2d(
                minimap_conv1,
                filters=32,
                kernel_size=4,
                strides=2,
                kernel_initializer=tf.glorot_normal_initializer(),
                # kernel_initializer=tf.orthogonal_initializer(),
                name="minimap_conv2",
                padding="same",
                activation=tf.nn.relu
        )
        
        screen_flatten = tf.layers.flatten(screen_conv2, name="screen_flatten")
        minimap_flatten = tf.layers.flatten(minimap_conv2, name="minimap_flatten")

        ns_fc = tf.layers.dense(self.ns, 256,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=tf.nn.tanh,
            name="ns_fc"
        )

        # fc_concat = tf.concat([screen_flatten, minimap_flatten, ns_fc], axis=1)
        fc_concat = screen_flatten

        state_representation = tf.layers.dense(fc_concat, 256,
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=tf.nn.relu,
            name="state_representation"
        )

        self.logits_pi = tf.layers.dense(state_representation, self.num_actions["base_action"],
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            # activation=tf.nn.softmax,
            activation=None,
            name="pi"
        )
        self.pi = tf.nn.softmax(self.logits_pi)

        self.v = tf.layers.dense(state_representation, 1,
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=None,
            name="vf"
        )

        self.args = dict()
        self.args_raw = dict()
        for act_type in actions.TYPES:
            if act_type.name in ("screen", "screen2", "minimap"):
                # arg_out = self._fc_spatial_argument(state_representation, act_type.name)
                raw_logits = self._non_spatial_argument(state_representation, Config.IMAGE_SIZE**2, act_type.name)
            else:
                raw_logits = self._non_spatial_argument(state_representation, act_type.sizes[0], act_type.name)
            self.args[act_type.name] = tf.nn.softmax(raw_logits)
            self.args_raw[act_type.name] = raw_logits

    def _fully_conv(self):
        screen_processed = self._embed_features(self.screen, features.SCREEN_FEATURES, "screen")
        minimap_processed = self._embed_features(self.minimap, features.MINIMAP_FEATURES, "minimap")
        screen_conv1 = tf.layers.conv2d(
            screen_processed,
            filters=16,
            kernel_size=5,
            strides=1,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="screen_conv1",
            padding="same",
            activation=tf.nn.relu
        )
        screen_conv2 = tf.layers.conv2d(
            screen_conv1,
            filters=24,
            kernel_size=3,
            strides=1,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="screen_conv2",
            padding="same",
            activation=tf.nn.relu
        )
        
        minimap_conv1 = tf.layers.conv2d(
            minimap_processed,
            filters=16,
            kernel_size=5,
            strides=1,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="minimap_conv1",
            padding="same",
            activation=tf.nn.relu
        )
        minimap_conv2 = tf.layers.conv2d(
            minimap_conv1,
            filters=24,
            kernel_size=3,
            strides=1,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="minimap_conv2",
            padding="same",
            activation=tf.nn.relu
        )

        ns_broadcast = tf.tile( tf.expand_dims(tf.expand_dims(self.ns, 1), 2), tf.stack([1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1]) )

        # state_representation = tf.concat([screen_conv2, minimap_conv2], axis=3)
        state_representation = tf.concat([screen_conv2, minimap_conv2, ns_broadcast], axis=3)
        state_representation_flattened = tf.layers.flatten(state_representation, name="state_representation_flattened")

        fc1 = tf.layers.dense(state_representation_flattened, 256,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=tf.nn.relu,
            name="fc1"
        )

        self.logits_pi = tf.layers.dense(fc1, self.num_actions["base_action"],
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            # activation=tf.nn.softmax,
            activation=None,
            name="pi"
        )
        self.pi = tf.nn.softmax(self.logits_pi)

        self.v = tf.layers.dense(fc1, 1,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=None,
            name="vf"
        )

        self.args = dict()
        self.args_raw = dict()
        for act_type in actions.TYPES:
            if act_type.name in ("screen", "screen2", "minimap"):
                raw_logits = self._fully_conv_spatial_argument(state_representation, act_type.name)
            else:
                raw_logits = self._non_spatial_argument(state_representation_flattened, act_type.sizes[0], act_type.name)
            self.args[act_type.name] = tf.nn.softmax(raw_logits)
            self.args_raw[act_type.name] = raw_logits

    def _create_graph(self):
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.screen = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, Config.NUM_SCREEN_FEATURES))
        self.minimap = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, Config.NUM_MINIMAP_FEATURES))
        self.ns = tf.placeholder(tf.float32, (None, Config.NUM_NONSPATIAL_FEATURES))
        self.act_mask = tf.placeholder(tf.float32, (None, self.num_actions["base_action"]))
        # self._atari()
        self._fully_conv()
        
       
        self.value = tf.squeeze(self.v, axis=1)
        self.acts = tf.placeholder(tf.int32, name="acts", shape=[None])
        self.act_args = {act_type.name: tf.placeholder(tf.int32, name="{}".format(act_type.name), shape=[None]) for act_type in actions.TYPES}
        self.act_args_used = {act_type.name: tf.placeholder(tf.float32, name="{}_used".format(act_type.name), shape=[None]) for act_type in actions.TYPES}
        # self.advs = tf.placeholder(tf.float32, [None])
        self.rewards = tf.placeholder(tf.float32, name="rewards", shape=[None])
        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        # acts_one_hot = tf.one_hot(self.acts, self.num_actions["base_action"])

        # valid_action_prob_sum = tf.reduce_sum(self.logits_pi * self.act_mask, axis=-1, keepdims=True)       # sum all valid non spatial action prob
        # masked_action_prob = (self.logits_pi * self.act_mask) / valid_action_prob_sum
        # neg_tensor = tf.cond(tf.equal(self.act_mask, tf.constant(1.0)), lambda: tf.constant(0), lambda: tf.constant(-999999.0))
        neg_tensor = tf.to_float(tf.equal(self.act_mask, tf.constant(0.0))) * tf.constant(-99999.0)
        action_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.acts, logits=self.logits_pi * self.act_mask + neg_tensor)
        for act_type in actions.TYPES:
            action_log_prob += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.act_args[act_type.name], logits=self.args_raw[act_type.name]) * self.act_args_used[act_type.name]
        

        cost_entropy = tf.reduce_sum(-tf.nn.softmax(self.logits_pi) * tf.nn.log_softmax(self.logits_pi), axis=-1)
        for act_type in actions.TYPES:
            cost_entropy += tf.reduce_sum(-tf.nn.softmax(self.args_raw[act_type.name]) * tf.nn.log_softmax(self.args_raw[act_type.name]), axis=-1) * self.act_args_used[act_type.name]
        cost_entropy = tf.reduce_sum(cost_entropy)
        


        '''
        # action log probability
        valid_action_prob_sum = tf.reduce_sum(self.pi * self.act_mask, axis=1)       # sum all valid non spatial action prob
        action_prob = tf.reduce_sum(self.pi * acts_one_hot, axis=1)
        action_log_prob = tf.log(tf.maximum(action_prob / valid_action_prob_sum, self.log_epsilon))
    
        for act_type in actions.TYPES:
            indexes = tf.stack([tf.range(tf.shape(self.act_args[act_type.name])[0]), self.act_args[act_type.name]], axis=1)
            arg_log_prob = tf.log(tf.maximum(tf.gather_nd(self.args[act_type.name], indexes), self.log_epsilon))
            action_log_prob += self.act_args_used[act_type.name] * arg_log_prob
        '''
        '''
        cost_entropy = tf.reduce_sum(-tf.reduce_sum(self.pi * tf.log(self.pi + 1e-8), axis=1))
        # args entropy
        
        for act_type in actions.TYPES:
            cost_entropy += tf.reduce_sum(-tf.reduce_sum(self.args[act_type.name] * tf.log(tf.maximum(self.args[act_type.name], self.log_epsilon)), axis=1) * self.act_args_used[act_type.name]) / tf.maximum(tf.reduce_sum(self.act_args_used[act_type.name]), 1.)
        '''

        '''
        entropy = -tf.reduce_sum(self.pi * tf.log(self.pi + 1e-8), axis=1)
        for act_type in actions.TYPES:
            entropy += -tf.reduce_sum(self.args[act_type.name] * tf.log(self.args[act_type.name] + 1e-8), axis=1) * self.act_args_used[act_type.name]
        
        cost_entropy = tf.reduce_sum(entropy, axis=0)
        '''

        cost_p = tf.reduce_sum((self.rewards - tf.stop_gradient(self.value)) * action_log_prob)
        cost_v = 0.5 * tf.reduce_sum(tf.square(self.rewards - self.value))
        loss = cost_p - cost_entropy * self.var_beta + cost_v * self.vl_coef
        self.cost_p = cost_p
        self.cost_v = cost_v
        self.loss = loss

        if Config.OPTIMIZER == "adam":
            self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        elif Config.OPTIMIZER == "rmsprop":
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                # decay=Config.RMSPROP_DECAY,
                # momentum=Config.RMSPROP_MOMENTUM,
                # epsilon=Config.RMSPROP_EPSILON
            )
        else:
            raise NotImplementedError
        
        self.grad_vars = self.opt.compute_gradients(loss)
        if Config.USE_GRAD_CLIP:
            grad = [x[0] for x in self.grad_vars]
            vars = [x[1] for x in self.grad_vars]
            grad, _ = tf.clip_by_global_norm(grad, Config.GRAD_CLIP_NORM)
            self.train_op = self.opt.apply_gradients(zip(grad, vars), global_step=self.global_step)
        else:
            self.train_op = self.opt.apply_gradients(self.grad_vars, global_step=self.global_step)

    def _embed_features(self, t, input_features, input_name):
        split_features = tf.split(t, len(input_features), -1)   # split the data along last dimension (channel)
        out = None
        map_list = []
        for idx, feature in enumerate(input_features):
            if feature.type == features.FeatureType.CATEGORICAL:
                with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                    embedding_table = tf.get_variable("{}_{}".format(input_name, feature.name),
                                        dtype=tf.float32,
                                        shape=[feature.scale, max(min(np.round(np.log2(feature.scale)), 10), 1)],
                                        initializer=tf.glorot_normal_initializer())
                    embedding_table = tf.concat((tf.zeros(shape=[1, max(min(np.round(np.log2(feature.scale)), 10), 1)]), embedding_table[1:, :]), 0)
                    # embedding_table = tf.nn.relu(embedding_table)
                    out = tf.nn.embedding_lookup(embedding_table, tf.to_int32(tf.squeeze(split_features[idx], -1)))
            elif feature.type == features.FeatureType.SCALAR:
                out = tf.log1p(split_features[idx])
            else:
                raise AttributeError
            map_list.append(out)
        processed_features = tf.concat(map_list, -1)
        print ("**** proccessed {} : {} ***".format(input_name, processed_features))
        return processed_features

    def _fc_spatial_argument(self, state, name):
        x = tf.layers.dense(state, self.img_size,
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.softmax,
            name="%s_x" % name
        )
        y = tf.layers.dense(state, self.img_size,
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.softmax,
            name="%s_y" % name
        )
        pos = tf.layers.flatten(tf.reshape(x, [-1, 1, Config.IMAGE_SIZE]) * tf.reshape(y, [-1, Config.IMAGE_SIZE, 1]))
        return pos

    def _non_spatial_argument(self, fc, size, name):
        temp = tf.layers.dense(fc, size,
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            # activation=tf.nn.softmax,
            activation=None,
            name="arg_%s" % name
        )
        # temp_softmax = (tf.nn.softmax(temp) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * size)
        # return temp_softmax
        return temp

    def _fully_conv_spatial_argument(self, state, name):
        temp = tf.layers.conv2d(
            state,
            filters=1,
            kernel_size=1,
            strides=1,
            kernel_initializer=tf.glorot_uniform_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=None,
            padding="same",
            name="arg_%s" % name
        )
        return tf.layers.flatten(temp)

    def _prepare_feed_dict(self, state_dict):
        feed_dict = {
            self.screen: state_dict["screen"],
            self.minimap: state_dict["minimap"],
            self.ns: state_dict["ns"],
            self.act_mask: state_dict["available_actions"]
        }
        return feed_dict

    def _prepare_actions(self, base_action, args):
        predict_actions = [dict() for _ in range(len(base_action))]
        for i in range(len(base_action)):
            predict_actions[i]["base_action"] = base_action[i]
            for act_type in actions.TYPES:
                predict_actions[i][act_type.name] = args[act_type.name][i]
        return predict_actions

    def predict_v(self, x):
        feed_dict = self._prepare_feed_dict(x)
        prediction = self.sess.run(self.value, feed_dict=feed_dict)
        return prediction

    def predict_p(self, x):
        feed_dict = self._prepare_feed_dict(x)
        pi, act_args = self.sess.run([self.pi, self.args], feed_dict=feed_dict)
        action_dict = self._prepare_actions(pi, act_args)
        return action_dict
    
    def predict_p_and_v(self, x):
        feed_dict = self._prepare_feed_dict(x)
        pi, act_args, v = self.sess.run([self.pi, self.args, self.value], feed_dict=feed_dict)
        action_dict = self._prepare_actions(pi, act_args)
        
        return action_dict, v

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def _extract_training_data(self, exps):
        screens, minimaps, nss, base_actions, avail_actions, rewards = [], [], [], [], [], []
        args, arg_used = dict(), dict()
        for act_type in actions.TYPES:
            args[act_type.name] = []
            arg_used[act_type.name] = []
        for exp in exps:
            screens.append(exp.state["screen"])
            minimaps.append(exp.state["minimap"])
            nss.append(exp.state["ns"])
            avail_actions.append(exp.state["available_actions"])
            base_actions.append(exp.action["base_action"])
            rewards.append(exp.reward)
            for act_type in actions.TYPES:
                args[act_type.name].append(exp.action[act_type.name] if exp.action[act_type.name] != -1 else 0)
                arg_used[act_type.name].append(float(exp.action[act_type.name] != -1))

        return screens, minimaps, nss, base_actions, avail_actions, rewards, args, arg_used

    def train(self, exps, trainer_id):
        screens, minimaps, nss, base_actions, avail_actions, rewards, args, arg_used = self._extract_training_data(exps)

        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({
            self.screen: screens,
            self.minimap: minimaps,
            self.ns: nss,
            self.acts: base_actions,
            self.act_mask: avail_actions,
            self.rewards: rewards
        })
        for act_type in actions.TYPES:
            feed_dict[self.act_args[act_type.name]] = args[act_type.name]
            feed_dict[self.act_args_used[act_type.name]] = arg_used[act_type.name]

        self.sess.run(self.train_op, feed_dict=feed_dict)

        # print ("Policy Loss:{0:.3f}, Value Loss:{1:.3f}".format(policy_loss, value_loss))

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)

    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)

    def _create_tensor_board(self):
        tf.summary.scalar("Pcost", self.cost_p)
        tf.summary.scalar("Vcost", self.cost_v)
        tf.summary.scalar("LearningRate", self.var_learning_rate)
        tf.summary.scalar("Beta", self.var_beta)
        tf.summary.histogram("activation_v", self.v)
        tf.summary.histogram("activation_p", self.pi)
        self.summary_op = tf.summary.merge_all()
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)
        '''        
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1))
        # summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        # summaries.append(tf.summary.histogram("activation_n1", self.n1))
        # summaries.append(tf.summary.histogram("activation_n2", self.n2))
        # summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.v))
        summaries.append(tf.summary.histogram("activation_p", self.pi))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)
        '''

    def log(self, exps):
        screens, minimaps, nss, base_actions, avail_actions, rewards, args, arg_used = self._extract_training_data(exps)
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({
            self.screen: screens,
            self.minimap: minimaps,
            self.ns: nss,
            self.acts: base_actions,
            self.act_mask: avail_actions,
            self.rewards: rewards
        })
        for act_type in actions.TYPES:
            feed_dict[self.act_args[act_type.name]] = args[act_type.name]
            feed_dict[self.act_args_used[act_type.name]] = arg_used[act_type.name]
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)