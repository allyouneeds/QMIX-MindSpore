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
"""calculate time"""
import time
import numpy as np


def print_time(start_time, T, t_max, episode, episode_rewards):
    time_elapsed = time.time() - start_time
    T = max(1, T)
    left_time = time_elapsed * (t_max - T) / T
    # Just in case its over 100 days
    left_time = min(left_time, 60 * 60 * 24 * 100)
    last_reward = "N\\A"
    if len(episode_rewards) > 5:
        last_reward = "{:.2f}".format(np.mean(episode_rewards[-50:]))
    print("\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, Reward: {}, \n\x1b[KElapsed: {}, Left: {}\n".format(
        episode, T, t_max, last_reward, time_str(time_elapsed), time_str(left_time)), " " * 10, end="\r")


def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_ = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_ = min(time_, 60 * 60 * 24 * 100)
    return time_str(time_)


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string
