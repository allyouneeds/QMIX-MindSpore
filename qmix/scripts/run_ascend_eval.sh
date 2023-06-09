#!/bin/bash
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

# an simple tutorial as follows, more parameters can be setting
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
Info(){
    echo "Failed!"
    echo "Usage: bash run_ascend_eval.sh [DEVICE] [MAP_NAME] [CKPT_PATH]"
    echo "Example: bash run_ascend_eval.sh 1 8m ./results/models/"
}

if [ $# != 3 ];then
    Info
    exit 1
fi
DEVICE_ID=$1
MAP_NAME=$2
CKPT_PATH=$3
python -s ${self_path}/../ascend_src/main.py --config=qmix --env-config=sc2 with env_args.map_name=$MAP_NAME device_id=$DEVICE_ID checkpoint_path=$CKPT_PATH evaluate=True> evallog.txt 2>&1 &
