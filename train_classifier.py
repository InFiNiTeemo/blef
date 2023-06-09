import os
import pandas as pd
import warnings

import zoo_transforms
from training.config import load_config
from training.losses import tn_score, tp_score

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120
import os
from typing import Dict

import numpy as np
import torch.distributed
import torch.distributed as dist
from sklearn.metrics import classification_report, precision_score, average_precision_score
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from tqdm import tqdm

from class_config import CLASSES_21
from metrics import bird_metric
from training.val_dataset import BirdDataset

warnings.filterwarnings("ignore")
import argparse

from training.trainer import Evaluator, PytorchTrainer, TrainConfiguration


class BirdEvaluator(Evaluator):
    def __init__(self, args, train_full=True) -> None:
        super().__init__()
        self.args = args
        self.train_full = train_full

    def init_metrics(self) -> Dict:
        return {"f1_score": 0, "lb": 0.}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:
        if self.train_full:
            return self.init_metrics()

        conf_name = os.path.splitext(os.path.basename(self.args.config))[0]
        val_dir = os.path.join(self.args.val_dir, conf_name, str(self.args.fold))
        os.makedirs(val_dir, exist_ok=True)

        ## TODO: thresholding?
        val_out = {"gts": [], "preds": []}

        for sample in tqdm(dataloader):
            wav = sample["wav"]
            labels = sample["labels"].numpy()

            outs = model(wav, is_test=True)
            outs = outs['logit'].sigmoid().cpu().detach().numpy()

            val_out['gts'].extend(labels)
            val_out['preds'].extend(outs)

        val_template = "{conf_name}_val_outs_{local_rank}.npy"
        val_out_path = os.path.join(val_dir, val_template.format(conf_name=conf_name, local_rank=local_rank))
        np.save(val_out_path, val_out)

        if distributed:
            dist.barrier()

        best_threshold = -1
        best_f1, best_lb = -1, -1
        if self.args.local_rank == 0:
            gts = []
            preds = []
            for rank in range(self.args.world_size):
                val_out_path = os.path.join(val_dir, val_template.format(conf_name=conf_name, local_rank=rank))
                outs = np.load(val_out_path, allow_pickle=True)
                gts.append(np.array(outs[()]['gts']))
                preds.append(np.array(outs[()]['preds']))
            gts = np.concatenate(gts, axis=0)
            preds = np.concatenate(preds, axis=0)
            #for threshold in np.arange(0.1, 0.9, 0.05):

            acc = average_precision_score(
                gts,
                preds,
                average='macro',
            )

            pad_3 = bird_metric.padded_cmap(gts, preds, 3)
            pad_5 = bird_metric.padded_cmap(gts, preds, 5)
            lb = pad_5
            f1s = bird_metric.get_f1(gts, preds)
            #print(classification_report(gts, preds > threshold, target_names=CLASSES_21))
            if lb > best_lb:
                best_lb = lb
                best_f1 = f1s

            print(f"=>pad_5:{pad_5:.4f},   pad_3:{pad_3:.4f},   acc:{acc:.4f}")

        if distributed:
            dist.barrier()
        empty_cache()
        return {"f1_score": best_f1, "lb": best_lb}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        for metric in ["f1_score", "lb"]:
            if current_metrics[metric] > prev_metrics[metric]:
                print("{} improved from {:.6f} to {:.6f}".format(metric, prev_metrics[metric], current_metrics[metric]))
                improved[metric] = current_metrics[metric]
            else:
                print("{} {:.6f} current {:.6f}".format(metric, prev_metrics[metric], current_metrics[metric]))
        return improved


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/v2s.json")
    arg('--workers', type=int, default=12, help='number of cpu threads to use PER GPU!')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='val_')
    arg('--val-dir', type=str, default="validation")
    arg('--data-dir', type=str, default="/kaggle/input/")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--fp16', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)
    arg("--val", action='store_true', default=False)
    arg("--freeze-bn", action='store_true', default=False)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    conf = load_config(args.config)
    train_period = conf["encoder_params"].get("duration") 
    infer_period = conf["encoder_params"].get("val_duration")

    print(f"""
    creating dataset for fold {args.fold}
    transforms                {conf.get("train_transforms")}
    train_period              {train_period}
    infer_period              {infer_period} 
    2nd_weight                {conf.get("secondary_weight")}
    """)

    train_transforms = zoo_transforms.__dict__[conf.get("train_transforms")]

    ## set 1 csv
    train_dataset = BirdDataset(mode="train", folds_csv=args.folds_csv, dataset_dir=args.data_dir, fold=args.fold,
                                multiplier=conf.get("multiplier", 1), duration=train_period, transforms=train_transforms,
                                n_classes=conf['encoder_params']['classes'],
                                secondary_weight=conf.get("secondary_weight", 1),)
    val_dataset = BirdDataset(mode="val", folds_csv=args.folds_csv, dataset_dir=args.data_dir, fold=args.fold, duration=infer_period,
                              n_classes=conf['encoder_params']['classes'],
                              secondary_weight=conf.get("secondary_weight", 1),)
    return train_dataset, val_dataset


def main():
    args = parse_args()
    conf = load_config(args.config)
    print(conf)
    trainer_config = TrainConfiguration(
        config_path=args.config,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        distributed=args.distributed,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score,
        fp16=args.fp16,
        freeze_bn=args.freeze_bn,
        mixup_prob=conf.get("mixup_prob", 0.5)
    )

    data_train, data_val = create_data_datasets(args)
    birds_evaluator = BirdEvaluator(args, args.fold not in [0, 1, 2, 3, 4])
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=birds_evaluator, fold=args.fold,
                             train_data=data_train, val_data=data_val)

    if args.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == '__main__':
    main()
