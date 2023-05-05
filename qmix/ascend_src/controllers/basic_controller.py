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
"""controllers"""
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import mindspore.ops as ops
import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import save_checkpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net


# This multi-agent controller shares parameters between agents
class BasicMAC(nn.Cell):
    def __init__(self, scheme, groups, args):
        super(BasicMAC, self).__init__()

        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.broadcast_to = ops.BroadcastTo((self.args.batch_size_run, self.n_agents, -1))
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self.build_inputs(ep_batch, t)
        self.hidden_states = self.broadcast_to(self.hidden_states)
        self.hidden_states = self.infer_agent.reshape_hidden(self.hidden_states)
        agent_outs, self.hidden_states = self.infer_agent(agent_inputs, self.hidden_states)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        expand_dims = ops.ExpandDims()
        broadcast_to = ops.BroadcastTo((batch_size, self.n_agents, -1))
        self.hidden_states = broadcast_to(expand_dims(self.agent.init_hidden(), 0))

    def parameters(self):
        return self.agent.trainable_params()

    def load_infer_state(self):
        par_dict = self.agent.parameters_dict()
        param_dict = {}
        for name in par_dict:
            parameter = par_dict[name]
            name = name.replace('qmix.netWithLoss.agent.', 'mac.infer_agent.', 1)
            param_dict[name] = parameter
        load_param_into_net(self.infer_agent, param_dict)

    def load_state(self, other_mac):
        par_dict = other_mac.agent.parameters_dict()
        param_dict = {}
        for name in par_dict:
            parameter = par_dict[name]
            name = name.replace('netWithLoss.agent.', 'network.target_agent.', 1)
            param_dict[name] = parameter
        load_param_into_net(self.agent, param_dict)

    def save_models(self, path):
        save_checkpoint(self.agent, "{}/agent.ckpt".format(path))

    def load_models(self, path):
        param_dict = load_checkpoint("{}/agent.ckpt".format(path))
        load_param_into_net(self.agent, param_dict)

    def load_target_models(self, path):
        par_dict = load_checkpoint("{}/agent.ckpt".format(path))
        param_dict = {}
        for name in par_dict:
            parameter = par_dict[name]
            name = name.replace('agent.', 'target_agent.', 1)
            param_dict[name] = parameter
        load_param_into_net(self.agent, param_dict)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.infer_agent = agent_REGISTRY['infer'](input_shape, self.args)

    def build_inputs(self, batch, t):
        # Assumes homogeneous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            zeroslike = ops.ZerosLike()
            if t == 0:
                inputs.append(zeroslike(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            eye = ops.Eye()
            expand_dims = ops.ExpandDims()
            broadcast_to = ops.BroadcastTo((bs, -1, -1))
            inputs.append(broadcast_to(expand_dims(eye(self.n_agents, self.n_agents, ms.float32), 0)))
        reshape = ops.Reshape()
        cat = ops.Concat(1)
        inputs = cat([reshape(x, (bs * self.n_agents, -1)) for x in inputs])
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
