
# https://www.kaggle.com/code/nischaydnk/split-creating-melspecs-stage-1
def add_path_df(df):
    df["path"] = [str(Config.audios_path / filename) for filename in df.filename]
    df = df.reset_index(drop=True)
    pool = joblib.Parallel(2)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.path]
    df2 = pd.DataFrame(pool(tqdm(tasks))).reset_index(drop=True)
    df = pd.concat([df, df2], axis=1).reset_index(drop=True)

    return df