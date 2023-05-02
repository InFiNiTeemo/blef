# from tensorboard.backend.event_processing import event_accumulator
#
# #加载日志数据
# ea=event_accumulator.EventAccumulator("logs/pretrain_TimmClassifier_tf_efficientnetv2_s_in21k_0")
# ea.Reload()
# print(vars(ea))
# print(ea.__dict__)
# print(ea.scalars.Keys())

# val_acc=ea.scalars.Items('val_acc')
# print(len(val_acc))
# print([(i.step,i.value) for i in val_acc])
#
# import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(6,4))
# ax1=fig.add_subplot(111)
# val_acc=ea.scalars.Items('val_acc')
# ax1.plot([i.step for i in val_acc],[i.value for i in val_acc],label='val_acc')
# ax1.set_xlim(0)
# acc=ea.scalars.Items('acc')
# ax1.plot([i.step for i in acc],[i.value for i in acc],label='acc')
# ax1.set_xlabel("step")
# ax1.set_ylabel("")
#
# plt.legend(loc='lower right')
# plt.show()

import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
for summary in summary_iterator("logs/pretrain_TimmClassifier_v1_eca_nfnet_l0_0/events.out.tfevents.1682352206.Mobius"):
    print(summary)