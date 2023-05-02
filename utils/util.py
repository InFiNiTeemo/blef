import os
import time
from pathlib import Path
from typing import Any, Union

import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# seq
def get_label_mappings(discourse_type):
    i_discourse_type = ['I-' + i for i in discourse_type]
    b_discourse_type = ['B-' + i for i in discourse_type]
    labels_to_ids = {k: v for v, k in enumerate(['O'] + i_discourse_type + b_discourse_type)}
    ids_to_labels = {k: v for v, k in labels_to_ids.items()}
    return labels_to_ids, ids_to_labels


def read_df(path: Union[Path, str])->pd.DataFrame:
    if isinstance(path, Path):
        path = str(path)

    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        return pd.read_json(path)

import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def feat_padding(input_ids, attention_mask, token_label, batch_length, padding_dict, padding_side):
    random_seed = None
    if padding_side == 'right':
        random_seed = 0
    elif padding_side == 'left':
        random_seed = 1
    else:
        random_seed = np.random.rand()

    # 剔除原有padding
    mask_index = attention_mask.nonzero().reshape(-1)
    input_ids = input_ids.index_select(0, mask_index)
    token_label = token_label.index_select(0, mask_index)
    attention_mask = attention_mask.index_select(0, mask_index)
    ids_length = len(input_ids)

    # 减去一部分长度
    if ids_length > batch_length:
        if random_seed <= 0.33:
            input_ids = input_ids[:batch_length]
            attention_mask = attention_mask[:batch_length]
            token_label = token_label[:batch_length]
        elif random_seed >= 0.66:
            input_ids = input_ids[-batch_length:]
            attention_mask = attention_mask[-batch_length:]
            token_label = token_label[-batch_length:]
        else:
            sub_length = ids_length - batch_length
            strat_idx = np.random.randint(sub_length + 1)
            input_ids = input_ids[strat_idx:strat_idx + batch_length]
            attention_mask = attention_mask[strat_idx:strat_idx + batch_length]
            token_label = token_label[strat_idx:strat_idx + batch_length]

    # 加上一部分长度
    if ids_length < batch_length:
        add_length = batch_length - ids_length
        if random_seed <= 0.33:
            input_ids = F.pad(input_ids, (0, add_length), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (0, add_length), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (0, add_length), "constant", padding_dict['input_ids'])
        elif random_seed >= 0.66:
            input_ids = F.pad(input_ids, (add_length, 0), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length, 0), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length, 0), "constant", padding_dict['input_ids'])
        else:
            add_length1 = np.random.randint(add_length + 1)
            add_length2 = add_length - add_length1
            input_ids = F.pad(input_ids, (add_length1, add_length2), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length1, add_length2), "constant",
                                   padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length1, add_length2), "constant", padding_dict['input_ids'])

    return input_ids, attention_mask, token_label

class Collate:
    def __init__(self, model_length=None, max_length=None, padding_side='right', padding_dict={}):
        self.model_length = model_length
        self.max_length = max_length
        self.padding_side = padding_side
        self.padding_dict = padding_dict

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_length = None
        if self.model_length is not None:
            batch_length = self.model_length
        else:
            batch_length = max([len(ids) for ids in output["input_ids"]])
            if self.max_length is not None:
                batch_length = min(batch_length, self.max_length)

        for i in range(len(output["input_ids"])):
            output_fill = feat_padding(output["input_ids"][i], output["attention_mask"][i], output["labels"][i],
                                       batch_length, self.padding_dict, padding_side=self.padding_side)
            output["input_ids"][i], output["attention_mask"][i], output["labels"][i] = output_fill

        # convert to tensors
        output["input_ids"] = torch.stack(output["input_ids"])
        output["attention_mask"] = torch.stack(output["attention_mask"])
        output["labels"] = torch.stack(output["labels"])

        return output



