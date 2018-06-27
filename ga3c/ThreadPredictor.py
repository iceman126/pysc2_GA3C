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

from threading import Thread

import numpy as np
import pickle

from Config import Config
from pysc2.lib import actions

class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        screens = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, Config.IMAGE_SIZE, Config.IMAGE_SIZE, Config.NUM_SCREEN_FEATURES),
            dtype=np.float32)
        minimaps = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, Config.IMAGE_SIZE, Config.IMAGE_SIZE, Config.NUM_MINIMAP_FEATURES),
            dtype=np.float32)
        nss = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, Config.NUM_NONSPATIAL_FEATURES),
            dtype=np.float32)
        avals = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, len(actions.FUNCTIONS)),
            dtype=np.float32)

        while not self.exit_flag:
            ids[0], state_dict = self.server.prediction_q.get()
            screens[0] = state_dict["screen"]
            minimaps[0] = state_dict["minimap"]
            nss[0] = state_dict["ns"]
            avals[0] = state_dict["available_actions"]

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                ids[size], state_dict = self.server.prediction_q.get()
                screens[size] = state_dict["screen"]
                
                minimaps[size] = state_dict["minimap"]
                nss[size] = state_dict["ns"]
                avals[size] = state_dict["available_actions"]
                size += 1

            batch = dict()
            batch["screen"] = screens[:size]
            batch["minimap"] = minimaps[:size]
            batch["ns"] = nss[:size]
            batch["available_actions"] = avals[:size]

            # print (batch["screen"][0, :, :, 4])

            # batch = states[:size]
            p, v = self.server.model.predict_p_and_v(batch)

            for i in range(size):
                if ids[i] < len(self.server.agents):
                    self.server.agents[ids[i]].wait_q.put((p[i], v[i]))