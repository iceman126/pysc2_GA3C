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

import numpy as np
from pysc2 import maps
from pysc2.env import sc2_env
from pysc2.lib import actions, features

class GameManager:
    def __init__(self, map_name, image_size):
        self.map_name = map_name
        self.image_size = image_size
        self.env = self._make_env(self.map_name, self.image_size)
        self.reset()

    def _make_env(self, map_name, image_size):
        agent_interface = features.parse_agent_interface_format(
            feature_screen=image_size,
            feature_minimap=image_size
        )
        env = sc2_env.SC2Env(map_name=map_name,
                             step_mul=8,
                             agent_interface_format=agent_interface,
                             visualize=False
        )
        return env

    def reset(self):
        observation, _ = GameManager._process_state(self.env.reset()[0])
        return observation

    def step(self, action):
        observation, info = GameManager._process_state(self.env.step(GameManager._construct_action(action, self.image_size))[0])
        return observation, info["reward"], info["done"], []

    @staticmethod
    def _construct_action(action, image_size):
        act_args = []
        for arg in actions.FUNCTIONS[action["base_action"]].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([int(action[arg.name] % image_size), int(action[arg.name] // image_size)])
            else:
                act_args.append([action[arg.name]])
        return [actions.FunctionCall(action["base_action"], act_args)]

    @staticmethod
    def _process_state(state):
        processed_screen = np.transpose(np.asarray(state.observation["feature_screen"]), [1, 2, 0])
        processed_minimap = np.transpose(np.asarray(state.observation["feature_minimap"]), [1, 2, 0])
        ns = np.asarray(state.observation["player"])

        np.set_printoptions(threshold=np.nan)

        # print (np.shape(processed_screen))
        # print (processed_screen)
        
        available_actions = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
        available_actions[np.asarray(state.observation["available_actions"])] = 1

        # print (available_actions)

        ob = {"screen": processed_screen, "minimap": processed_minimap, "ns": ns, "available_actions": available_actions}
        info = {"reward": state.reward, "done": state.last()}

        return ob, info