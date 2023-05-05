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
"""select action"""
import copy
from mindspore import Tensor
import mindspore as ms
import numpy as np
from .epsilon_schedules import DecayThenFlatSchedule


REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0
        # mask actions that are excluded from selection
        masked_q_values = copy.deepcopy(agent_inputs)
        if self.args.evaluate_310:
            masked_q_values = masked_q_values.asnumpy()
            avail_actions_np = avail_actions.asnumpy()
            masked_q_values[avail_actions_np == 0.0] = -float("inf")  # should never be selected!
            masked_q_values = Tensor(masked_q_values, ms.float32)
        else:
            masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        shape = agent_inputs[:, :, 0].shape
        random_numbers = Tensor(np.random.uniform(0, 1, shape), dtype=ms.float32)
        pick_random = (random_numbers < self.epsilon).astype("int32")
        random_actions = []
        avail_actions = avail_actions.asnumpy()
        for i in range(avail_actions.shape[1]):
            pro = avail_actions[0, i, :].astype(np.float32) / avail_actions[0, i, :].sum()
            random_actions.append(np.random.choice(range(avail_actions.shape[2]), 1, p=pro)[0])
        random_actions = Tensor([random_actions], ms.int32)
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.argmax(axis=2)
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
