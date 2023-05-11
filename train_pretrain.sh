#!/bin/bash

NUM_GPUS=$1
VAL_OUT_DIR=$2
CONFIG=$3
FOLD=$4
DATA_DIR=$5

PYTHONPATH=.  python -u -m torch.distributed.launch  --nproc_per_node=$NUM_GPUS  --master_port 9979  train_pretrain.py  \
 --world-size $NUM_GPUS  --config configs/${CONFIG}.json  --workers 8 --test_every 1 \
 --val-dir $VAL_OUT_DIR  --prefix pretrain_  --fold $FOLD --freeze-epochs 0 --fp16 --data-dir $DATA_DIR --test_every 1000 \
#  --resume weights/pretrain_TimmClassifier_v3_tf_efficientnet_b0_0_last

# cd ~/code/kaggle/BLEF/rank1/birdclef-2022/
# sh train_pretrain.sh 1 0 v2s 0 /kaggle/input
# tensorboard --logdir logs/pretrain_TimmClassifier_eca_nfnet_l0_0
# tensorboard --logdir logs/pretrain_TimmClassifier_v1_eca_nfnet_l0_0

# sh train_pretrain.sh 1 0 pre_nf0 0 /kaggle/input




#sh train_pretrain.sh 1 0 cls 0 /kaggle/input

# ef
# sh train_pretrain.sh 1 0 pre_ef 0 /kaggle/input
# tensorboard --logdir logs/pretrain_TimmClassifier_v3_tf_efficientnet_b0_0

# efv2
# sh train_pretrain.sh 1 0 pre_ef2 0 /kaggle/input