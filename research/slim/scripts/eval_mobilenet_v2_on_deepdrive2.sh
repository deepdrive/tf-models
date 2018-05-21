#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
#
# This script performs the following operations:
# 2. Fine-tunes Mobilenet v2 on the Deepdrive train set.
# 3. Evaluates the model on the Deepdrive eval set.
#
# Usage:
# scripts/finetune_mobilenet_v2_on_deepdrive.sh
#
# Should see steering error of about 0.1135 / Original Deepdrive 2.0 baseline steering error eval was ~0.2, train steering error: ~0.08
set -ev

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/a/mnet-test/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/a/mnet-test/models

# Where the dataset is saved to.
DATASET_DIR=/media/a/data-ext4/deepdrive-data

python eval_image_classifier.py \
  --dataset_name=deepdrive \
  --dataset_split_name=eval \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --eval_image_size=224

