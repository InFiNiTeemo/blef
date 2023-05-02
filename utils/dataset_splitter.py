import numpy as np
import pandas as pd

class KFold(object):
    """
    KFold: Group split by group_col or random_split
    """

    def __init__(self, random_seed, k_folds=10, flag_name='kfold'):
        self.k_folds = k_folds
        self.flag_name = flag_name
        np.random.seed(random_seed)

    def group_split(self, train_df, group_col):  # 同一个group会被 均分 到不同的folds
        group_value = list(set(train_df[group_col]))
        group_value.sort()
        fold_flag = [i % self.k_folds for i in range(len(group_value))]
        np.random.shuffle(fold_flag)
        train_df = train_df.merge(pd.DataFrame({group_col: group_value, self.flag_name: fold_flag}), how='left',
                                  on=group_col)
        return train_df

    def random_split(self, train_df) -> pd.DataFrame:
        fold_flag = [i % self.k_folds for i in range(len(train_df))]
        np.random.shuffle(fold_flag)
        train_df[self.flag_name] = fold_flag
        return train_df

    def stratified_split(self, train_df, group_col):  # stratify and split， 同一个group会被 划分 到相同的fold
        train_df[self.flag_name] = 1
        train_df[self.flag_name] = train_df.groupby(by=[group_col])[self.flag_name].rank(ascending=True,
                                                                                         method='first').astype(int)
        train_df[self.flag_name] = train_df[self.flag_name].sample(frac=1.0).reset_index(drop=True)
        train_df[self.flag_name] = train_df[self.flag_name] % self.k_folds
        return train_df


def split_dataset(df, frac=0.9, logger=None):
    if logger is not None:
        logger.info("**split dataset**")
    if isinstance(df, dict):
        return dict(df.items()[:int(len(df) * frac)]), dict(df.items()[int(len(df) * frac):])
    elif isinstance(df, list):
        return df[:int(len(df) * frac)], df[int(len(df) * frac):]
    if logger is not None:
        logger.info("*split dataframe*")
    index = df.sample(frac=frac).index
    train_df = df[df.index.isin(index)]
    val_df = df[~df.index.isin(index)]
    return train_df, val_df
