import pandas as pd
from sklearn import model_selection


def get_oof_ch_mean(df: pd.DataFrame):
    kf = model_selection.KFold(n_splits=8, random_state=99)
    ret_list = []
    for fold_index, oof_index in kf.split(df):
        #df.iloc[oof_index] = get_channel_mean(df.iloc[fold_index], df.iloc[oof_index])
        ret_list.append(get_channel_mean(df.iloc[fold_index], df.iloc[oof_index]))
    ret_df = pd.DataFrame(pd.concat(ret_list))
    return ret_df


def get_channel_mean(fold_df: pd.DataFrame, oof_df: pd.DataFrame):
    grouped = fold_df.groupby("channel")["is_attributed"].agg({"mean", "count"}).reset_index()
    grouped.columns = ["channel", "ch_mean", "ch_count"]
    ret_df = pd.merge(oof_df, grouped, on="channel", how="left")
    ret_df["ip_ch_mean"] = ret_df.groupby("ip")["ch_mean"].transform("mean")
    ret_df["ip_ch_count"] = ret_df.groupby("ip")["ch_count"].transform("mean")

    # grouped2 = fold_df.groupby("ip")["is_attributed"].agg({"mean", "count"}).reset_index()
    # grouped2.columns = ["ip", "ip_mean", "ip_count"]
    # ret_df = pd.merge(ret_df, grouped2, on="ip", how="left")

    # ret_df["app_ch_mean"] = ret_df.groupby("app")["mean"].transform("mean")
    # ret_df["app_ch_count"] = ret_df.groupby("app")["count"].transform("mean")
    # ret_df["os_ch_mean"] = ret_df.groupby("os")["mean"].transform("mean")
    # ret_df["os_ch_count"] = ret_df.groupby("os")["count"].transform("mean")
    return ret_df


def get_holdout_channel_mean(df: pd.DataFrame, holdout_df: pd.DataFrame):
    grouped = df.groupby("channel")["is_attributed"].agg({"mean", "count"}).reset_index()
    grouped.columns = ["channel", "ch_mean", "ch_count"]
    holdout_df = pd.merge(holdout_df, grouped, on="channel", how="left")
    holdout_df["ip_ch_mean"] = holdout_df.groupby("ip")["ch_mean"].transform("mean")
    holdout_df["ip_ch_count"] = holdout_df.groupby("ip")["ch_count"].transform("mean")

    # holdout_df["app_ch_mean"] = holdout_df.groupby("app")["mean"].transform("mean")
    # holdout_df["app_ch_count"] = holdout_df.groupby("app")["count"].transform("mean")
    # holdout_df["os_ch_mean"] = holdout_df.groupby("os")["mean"].transform("mean")
    # holdout_df["os_ch_count"] = holdout_df.groupby("os")["count"].transform("mean")

    # grouped2 = df.groupby("ip")["is_attributed"].agg({"mean", "count"}).reset_index()
    # grouped2.columns = ["ip", "ip_mean", "ip_count"]
    # holdout_df = pd.merge(holdout_df, grouped2, on="ip", how="left")
    return holdout_df


def get_additional_fe(df: pd.DataFrame):
    df["hourly_ip_ch_mean"] = df.groupby(["ip", "hour"])["ip_ch_mean"].transform("mean")
    df["hourly_ip_ch_count"] = df.groupby(["ip", "hour"])["ip_ch_count"].transform("mean")
    return df
