# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""infer net"""
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor, nn


class InferAgent(nn.Cell):
    def __init__(self, input_shape, args):
        super(InferAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Dense(input_shape, args.rnn_hidden_dim,
                            weight_init='uniform', bias_init='uniform')
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Dense(args.rnn_hidden_dim, args.n_actions,
                            weight_init='uniform', bias_init='uniform')
        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()
        self.h_in_shape = self.args.rnn_hidden_dim

    def init_hidden(self):
        # make hidden states on same device as model
        return Tensor(np.zeros([1, self.args.rnn_hidden_dim]), self.fc1.weight.dtype)

    def reshape_hidden(self, hidden_state):
        h_in = self.reshape(hidden_state, (-1, self.h_in_shape))
        return h_in

    def construct(self, inputs, h_in):
        fc1_out = self.fc1(inputs)
        x = self.relu(fc1_out)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
