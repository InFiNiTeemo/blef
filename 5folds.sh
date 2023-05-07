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


# python
# python utils/kaggle/submission_time.py


sh train_single.sh 0 ./outs cls_ef_v0 0 /kaggle/input
