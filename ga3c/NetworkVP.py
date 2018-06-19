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

        self.rand_num = np.random.rand(32,)

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

                # if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    def _create_graph(self):
        self.screen = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, Config.NUM_SCREEN_FEATURES))
        self.minimap = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, Config.NUM_MINIMAP_FEATURES))
        self.ns = tf.placeholder(tf.float32, (None, Config.NUM_NONSPATIAL_FEATURES))
        self.act_mask = tf.placeholder(tf.float32, (None, self.num_actions["base_action"]))
        screen_processed = self._embed_features(self.screen, features.SCREEN_FEATURES, "screen")
        self.screen_processed_test = screen_processed
        minimap_processed = self._embed_features(self.minimap, features.MINIMAP_FEATURES, "minimap")
        screen_conv1 = tf.layers.conv2d(
                screen_processed,
                filters=16,
                kernel_size=8,
                strides=4,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name="screen_conv1",
                padding="same",
                activation=tf.nn.relu
        )
        screen_conv2 = tf.layers.conv2d(
                screen_conv1,
                filters=32,
                kernel_size=4,
                strides=2,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name="screen_conv2",
                padding="same",
                activation=tf.nn.relu
        )
        minimap_conv1 = tf.layers.conv2d(
                minimap_processed,
                filters=16,
                kernel_size=8,
                strides=4,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name="minimap_conv1",
                padding="same",
                activation=tf.nn.relu
        )
        minimap_conv2 = tf.layers.conv2d(
                minimap_conv1,
                filters=32,
                kernel_size=4,
                strides=2,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name="minimap_conv2",
                padding="same",
                activation=tf.nn.relu
        )
        
        screen_flatten = tf.layers.flatten(screen_conv2, name="screen_flatten")
        minimap_flatten = tf.layers.flatten(minimap_conv2, name="minimap_flatten")

        ns_fc = tf.layers.dense(self.ns, 256,
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.tanh,
            name="ns_fc"
        )

        fc_concat = tf.concat([screen_flatten, minimap_flatten, ns_fc], axis=1)

        state_representation = tf.layers.dense(fc_concat, 256,
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            name="state_representation"
        )

        self.pi = tf.layers.dense(state_representation, self.num_actions["base_action"],
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.softmax,
            name="pi"
        ) 

        v = tf.layers.dense(state_representation, 1,
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=None,
            name="vf"
        )

        self.args = dict()
        for act_type in actions.TYPES:
            if act_type.name in ("screen", "screen2", "minimap"):
                arg_out = self._fc_spatial_argument(state_representation, act_type.name)
            else:
                arg_out = self._non_spatial_argument(state_representation, act_type.sizes[0], act_type.name)
            self.args[act_type.name] = arg_out

        self.value = tf.squeeze(v, axis=[1])
        self.acts = tf.placeholder(tf.int32, name="acts", shape=[None])
        self.act_args = {act_type.name: tf.placeholder(tf.int32, name="{}".format(act_type.name), shape=[None]) for act_type in actions.TYPES}
        self.act_args_used = {act_type.name: tf.placeholder(tf.float32, name="{}_used".format(act_type.name), shape=[None]) for act_type in actions.TYPES}
        # self.advs = tf.placeholder(tf.float32, [None])
        self.rewards = tf.placeholder(tf.float32, name="rewards", shape=[None])
        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        acts_one_hot = tf.one_hot(self.acts, self.num_actions["base_action"])

        # action log probability
        valid_action_prob_sum = tf.reduce_sum(self.pi * self.act_mask, axis=1)       # sum all valid non spatial action prob
        action_prob = tf.reduce_sum(self.pi * acts_one_hot, axis=1)
        action_log_prob = tf.log(action_prob / valid_action_prob_sum + 1e-8)

        neglogpac = action_log_prob
    
        for act_type in actions.TYPES:
            indexes = tf.stack([tf.range(tf.shape(self.act_args[act_type.name])[0]), self.act_args[act_type.name]], axis=1)
            arg_log_prob = tf.log(tf.gather_nd(self.args[act_type.name], indexes) + 1e-8)
            neglogpac += self.act_args_used[act_type.name] * arg_log_prob

        action_entropy = tf.reduce_mean(-tf.reduce_sum(self.pi * tf.log(self.pi + 1e-8), axis=1))

        # args entropy
        entropy = action_entropy
        for act_type in actions.TYPES:
            entropy += tf.reduce_sum(-tf.reduce_sum(self.args[act_type.name] * tf.log(self.args[act_type.name] + 1e-8), axis=1) * self.act_args_used[act_type.name]) / tf.maximum(tf.reduce_sum(self.act_args_used[act_type.name]), 1.)

        # self.pg_loss = -tf.reduce_mean((self.rewards - tf.stop_gradient(self.value)) * neglogpac)
        self.pg_loss = -tf.reduce_mean((self.rewards - tf.stop_gradient(self.value)) * neglogpac)
        self.policy_log = neglogpac
        # self.vf_loss = tf.reduce_mean(tf.square(self.value - self.rewards) / 2.)
        self.vf_loss = 0.5 * tf.reduce_mean(tf.square(self.rewards - self.value))
        self.loss = self.pg_loss - entropy * self.var_beta + self.vf_loss * self.vl_coef

        self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        self.grad_vars = self.opt.compute_gradients(self.loss)
        if Config.USE_GRAD_CLIP:
            grad = [x[0] for x in self.grad_vars]
            vars = [x[1] for x in self.grad_vars]
            grad, _ = tf.clip_by_global_norm(grad, Config.GRAD_CLIP_NORM)
            self.train_op = self.opt.apply_gradients(zip(grad, vars))
        else:
            self.train_op = self.opt.apply_gradients(self.grad_vars)

    def _embed_features(self, t, input_features, input_name):
        split_features = tf.split(t, len(input_features), -1)   # split the data along last dimension (channel)
        # print ("*** {} split_features: {} ***".format(input_name, split_features))
        out = None
        map_list = []
        for idx, feature in enumerate(input_features):
            if feature.type == features.FeatureType.CATEGORICAL:
                with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                    embedding_table = tf.get_variable("{}_{}".format(input_name, feature.name),
                                        dtype=tf.float32,
                                        shape=[feature.scale, max(min(np.round(np.log2(feature.scale)), 10), 1)],
                                        initializer=tf.glorot_normal_initializer())
                    # squeezed_feature = tf.squeeze(split_features[idx])
                    # out = tf.nn.embedding_lookup(embedding_table, tf.to_int32(tf.reshape(split_features[idx], (-1, Config.IMAGE_SIZE, Config.IMAGE_SIZE))))
                    out = tf.nn.embedding_lookup(embedding_table, tf.to_int32(tf.squeeze(split_features[idx], -1)))

                # print ("{}_{} {}".format(input_name, feature.name, out))

                '''
                dims = np.round(np.log2(feature.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                one_hot_maps = tf.one_hot(tf.to_int32(tf.squeeze(split_features[idx], -1)), feature.scale)
                out = tf.layers.conv2d(one_hot_maps,
                    filters=dims,
                    kernel_size=1,
                    strides=1,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    padding="same",
                    activation=tf.nn.relu
                )
                '''
            elif feature.type == features.FeatureType.SCALAR:
                out = tf.log1p(split_features[idx])
                # print ("{}_{} {}".format(input_name, feature.name, out))
            else:
                raise AttributeError
            map_list.append(out)
        processed_features = tf.concat(map_list, -1)
        print ("**** proccessed {} shape: {} ***".format(input_name, tf.shape(processed_features)))
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
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.softmax,
            name="arg_%s" % name
        )
        return temp

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

    def train(self, exps, trainer_id):
        screens = []
        minimaps = []
        nss = []
        base_actions = []
        avail_actions = []
        rewards = []
        args = dict()
        arg_used = dict()
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
                args[act_type.name].append(exp.action[act_type.name])
                arg_used[act_type.name].append(float(exp.action[act_type.name] != -1))


        # print ("Screen shape: {}".format(np.shape(screens)))
        # print ("Minimap shape: {}".format(np.shape(minimaps)))
        # print ("Ns shape: {}".format(np.shape(nss)))
        # print ("Acts shape: {}".format(np.shape(base_actions)))

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

        _, policy_log, v = self.sess.run([self.train_op, self.policy_log, self.value], feed_dict=feed_dict)

        # print (policy_log)
        # print (rewards)
        # print (v)

        '''        
        numbers = [0.27, 0.34, 0.16, 0.48]
        indexes = [0, 5, 20, 30]        

        for i in range(4):
            temp = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            for c in range(4):
                temp += screen_proccessed[indexes[i], :, :, c + 5] * numbers[c]
            
            temp_num = 0.0
            for r in range(Config.IMAGE_SIZE):
                temp_num += np.sum(temp[r])  * self.rand_num[r]

            print ("****** {}: {}".format(i, temp_num))
        '''
        

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

        # print (x)
        # print (y_r)
        # print (a)
        # feed_dict = self.__get_base_feed_dict()
        # feed_dict.update({self.screen: x, self.y_r: y_r, self.action_index: a})
        # self.sess.run(self.train_op, feed_dict=feed_dict)

'''
class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # As implemented in A3C paper
        self.n1 = self.conv2d_layer(self.x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.n2 = self.conv2d_layer(self.n1, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        _input = self.n2

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1 = self.dense_layer(self.flat, 256, 'dense1')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)
        if Config.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

            self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                        * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                      self.softmax_p, axis=1)
        
        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)
        
        if Config.DUAL_RMSPROP:
            self.opt_p = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

            self.opt_v = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.cost_all = self.cost_p + self.cost_v
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v)
                self.opt_grad_v_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v) 
                                            for g,v in self.opt_grad_v if not g is None]
                self.train_op_v = self.opt_v.apply_gradients(self.opt_grad_v_clipped)
            
                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p)
                self.opt_grad_p_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_p if not g is None]
                self.train_op_p = self.opt_p.apply_gradients(self.opt_grad_p_clipped)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.opt_grad = self.opt.compute_gradients(self.cost_all)
                self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
                self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_v = self.opt_p.minimize(self.cost_v, global_step=self.global_step)
                self.train_op_p = self.opt_v.minimize(self.cost_p, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)


    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.n1))
        summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})
    
    def train(self, x, y_r, a, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

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
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))

'''