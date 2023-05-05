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
"""onehot"""
import mindspore as ms
from mindspore import Tensor
import numpy as np


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        shape = list(tensor.shape)
        shape[3] = self.out_dim
        one_hot_label = np.zeros(shape, dtype=np.float32)
        for k in range(tensor.shape[1]):
            one_hot_label[0, k, :, :] = [[int(i == int(tensor[0][k][j][0])) for i in range(self.out_dim)] for j in
                                         range(tensor.shape[2])]
        one_hot_label = Tensor(one_hot_label, dtype=ms.float32)
        return one_hot_label

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), ms.float32
