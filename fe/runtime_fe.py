import pandas as pd
from sklearn import model_selection


def get_oof_ch_mean(df: pd.DataFrame):
    kf = model_selection.KFold(n_splits=4, random_state=99)
    ret_list = []
    for fold_index, oof_index in kf.split(df):
        df[oof_index] = get_channel_mean(df[fold_index], df[oof_index])
        #ret_list.append(get_channel_mean(df[fold_index], df[oof_index]))
    #ret_df = pd.DataFrame(pd.concat(ret_list))
    #return ret_df


def get_channel_mean(fold_df: pd.DataFrame, oof_df: pd.DataFrame):
    grouped = fold_df.groupby("channel")["is_attributed"].agg({"mean", "count"}).reset_index()
    oof_df = pd.merge(oof_df, grouped, on="channel", how="left")
    return oof_df


def get_holdout_channel_mean(df: pd.DataFrame, holdout_df: pd.DataFrame):
    grouped = df.groupby("channel")["is_attributed"].agg({"mean", "count"}).reset_index()
    holdout_df = pd.merge(holdout_df, grouped, on="channel", how="left")
    return holdout_df