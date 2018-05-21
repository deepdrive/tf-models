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
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# scripts/finetune_mobilenet_v2_on_deepdrive_fast.sh

# Should see steering error of about 0.1135 / Original Deepdrive 2.0 baseline steering error eval was ~0.2, train steering error: ~0.08
set -ev

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/a/mnet-test/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/a/mnet-test/models

# Where the dataset is saved to.
DATASET_DIR=/media/a/data-ext4/deepdrive-data

### Fine-tune only the new layers
python train_image_classifier.py \
  --dataset_name=deepdrive \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --train_image_size=224 \
  --checkpoint_path=/home/a/mnet-test/checkpoints/mobilenet_v2_1.0_224.ckpt \
  --checkpoint_exclude_scopes=MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics \
  --trainable_scopes=MobilenetV2/Logits,MobilenetV2/Predictions,MobilenetV2/predics \
  --max_number_of_steps=2000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=10 \
  --save_summaries_secs=60 \
  --log_every_n_steps=20 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

python eval_image_classifier.py \
  --dataset_name=deepdrive \
  --dataset_split_name=eval \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --eval_image_size=224

# Fine-tune all layers
python train_image_classifier.py \
  --dataset_name=deepdrive \
  --resume_deepdrive \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --train_image_size=224 \
  --max_number_of_steps=49147 \
  --batch_size=16 \
  --learning_rate=0.00004 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=180 \
  --save_summaries_secs=60 \
  --log_every_n_steps=20 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


python eval_image_classifier.py \
  --dataset_name=deepdrive \
  --dataset_split_name=eval \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_deepdrive \
  --eval_image_size=224

