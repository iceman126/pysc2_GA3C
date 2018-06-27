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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time
import pickle

from Config import Config
from Environment import Environment
from Experience import Experience

from pysc2.lib import actions, features

class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = experiences[t].reward
            if Config.REWARD_CLIPPING:
                r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)        # clip the rewards
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]                                                             # discard the last exp (last exp belongs to next batch)

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()                    # block the process and wait for return values
        return p, v

    def select_action(self, prediction, act_mask):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = dict()
            prediction["base_action"] /= np.sum(np.multiply(prediction["base_action"], act_mask), axis=-1, keepdims=True)
            prediction["base_action"] *= act_mask
            action["base_action"] = np.random.choice(self.num_actions["base_action"], p=prediction["base_action"])
            for act_type in actions.TYPES:
                action[act_type.name] = -1
            for arg in actions.FUNCTIONS[action["base_action"]].args:
                action[arg.name] = np.random.choice(np.prod(self.num_actions[arg.name]), p=prediction[arg.name])
        return action

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                raise Exception("Current state should not be None")
                self.env.step(0)  # 0 == NOOP
                continue

            prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction, self.env.current_state["available_actions"])
            reward, done = self.env.step(action)
            reward_sum += reward
            exp = Experience(self.env.previous_state, action, prediction, reward, done)
            if self.env.previous_state == None:
                print ("Previous state is None.")
            
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value
                # print ("Is Done: {}, Terminal reward: {}".format(done, terminal_reward))

                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                indexes = np.argpartition(np.array(updated_exps[-1].prediction["base_action"]), -3)[-3:]
                if Config.DEBUG:
                    print ("{0}\tprob: {1:.4f}, screen: {5}, minimap:{6}, {3}, {4},value: {2:.4f}".format(updated_exps[-1].action["base_action"], \
                        updated_exps[-1].prediction["base_action"][updated_exps[-1].action["base_action"]], updated_exps[-1].reward, \
                        indexes, updated_exps[-1].prediction["base_action"][indexes], updated_exps[-1].action["screen"], updated_exps[-1].action["minimap"]
                    ))
                yield updated_exps, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            # for x_, r_, a_, reward_sum in self.run_episode():
            for exps, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(exps) + 1  # +1 for last frame that we drop
                # if Config.DEBUG:
                #     print ("Agent: Put episode into queue. Episode Length:{}".format(total_length))
                self.training_q.put(exps)
            self.episode_log_q.put((datetime.now(), total_reward, total_length))
