import copy

import pandas as pd

from utils.dataset_splitter import KFold
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

from audio.process_audio import get_audios_as_images


data = pd.read_csv("folds.csv")
# 修正class过少的
class_counts = data['primary_label'].value_counts()
low_count_classes = class_counts[class_counts < 2].index.tolist()
print(low_count_classes)

data.to_csv("folds.csv", index=False)

data.loc[data["primary_label"].isin(low_count_classes), "fold"] = 4
get_audios_as_images(data)

data.to_csv("folds.csv", index=False)

# data.drop("fold", axis=1, inplace=True)
# data = kfold.stratified_split(data, "primary_label").rename(columns={"kfold": "fold"})
# print(data.head(30))
#
# get_audios_as_images(data)
# data.to_csv("folds.csv", index=False)
