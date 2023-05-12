#GPU=$1
#VAL_OUT_DIR=$2
#CONFIG=$3
#FOLD=$4
#DATA_DIR=$5
#PREFIX=$6
#RESUME=$7


# bce loss 0.603   0.84
# sh train_single.sh 0 ./outs cls_nf0_v5 1 /kaggle/input nf0_v5_ pretrain_TimmClassifier_eca_nfnet_l0_0_last

# 1 way  acc 0.581
# sh train_single.sh 0 ./outs cls_nf0_v6 0 /kaggle/input nf0_v6_ pretrain_TimmClassifier_eca_nfnet_l0_0_last
# sh train_single.sh 0 ./outs cls_nf0_v6 1 /kaggle/input nf0_v6_ pretrain_TimmClassifier_eca_nfnet_l0_0_last
#sh train_single.sh 0 ./outs cls_nf0_v6 2 /kaggle/input nf0_v6_ pretrain_TimmClassifier_eca_nfnet_l0_0_last
#sh train_single.sh 0 ./outs cls_nf0_v6 3 /kaggle/input nf0_v6_ pretrain_TimmClassifier_eca_nfnet_l0_0_last
#sh train_single.sh 0 ./outs cls_nf0_v6 4 /kaggle/input nf0_v6_ pretrain_TimmClassifier_eca_nfnet_l0_0_last
#


# 2 way
# sh train_single.sh 0 ./outs cls_nf0_v7 1 /kaggle/input nf0_v7_ pretrain_TimmClassifier_eca_nfnet_l0_0_last


# cd ~/code/kaggle/BLEF/rank1/birdclef-2022/
# tensorboard --logdir logs/nf0_v6_TimmClassifier_eca_nfnet_l0_0


# ef

# sh train_single.sh 0 ./outs cls_ef_v0 0 /kaggle/input


# matrix
# acc f1 lb

# e20
# baseline
# 59.0 f1(54.8) 84.8
# sh train_single.sh 0 ./outs cls_ef_v0 0 /kaggle/input ef_v0 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20

# set1
# 58.0 f1(55.6) 84.83
# sh train_single.sh 0 ./outs cls_ef_v1 0 /kaggle/input ef_v1_e20_ pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20

# cut mix 0.2
# 59.2 52.8 84.7
#sh train_single.sh 0 ./outs cls_ef_v2 0 /kaggle/input ef_v2_e20_ pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20

# e20 primary
# best leaderboard
# 0.81 for 5 folds
# 63.9 57.9 89.6
#sh train_single.sh 0 ./outs cls_ef_v0 0 /kaggle/input ef_v0 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v0 1 /kaggle/input ef_v0 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v0 2 /kaggle/input ef_v0 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v0 3 /kaggle/input ef_v0 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v0 4 /kaggle/input ef_v0 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20


# use set3
# 65.3, 58.3, 90.0
#sh train_single.sh 0 ./outs cls_ef_v3 0 /kaggle/input ef0_v3 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20

# use epoch 10 有空交一发
#=>pad_5:0.8915,   pad_3:0.8595,   acc:0.6339
#f1_score 0.578581 current 0.573044
#lb improved from 0.890443 to 0.891461
#sh train_single.sh 0 ./outs cls_ef_v0 0 /kaggle/input ef0_v0_e10 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e10

# use different font-ends     https://www.kaggle.com/competitions/birdclef-2022/discussion/327193
#sh train_single.sh 0 ./outs cls_ef_v4 0 /kaggle/input ef0_v4 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v4 1 /kaggle/input ef0_v4 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v4 2 /kaggle/input ef0_v4 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v4 3 /kaggle/input ef0_v4 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20
#sh train_single.sh 0 ./outs cls_ef_v4 4 /kaggle/input ef0_v4 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0__e20




# 5.12
# sh train_pretrain.sh 1 0 pre_ef1 0 /kaggle/input

# sh train_pretrain.sh 1 0 pre_ef 0 /kaggle/input


# e40 bad
# sh train_single.sh 0 ./outs cls_ef_v0 0 /kaggle/input ef_v0 pretrain_TimmClassifier_v3_tf_efficientnet_b0_0_e40



## ** ef2 ** ##
# no 2023 data pretrain
# 61.17, 56.07, 88.2
# sh train_single.sh 0 ./outs cls_ef2_v0 0 /kaggle/input pretrain_TimmClassifier_v3_tf_efficientnetv2_s_in21k_0_e20




# python
# python utils/kaggle/submission_time.py



