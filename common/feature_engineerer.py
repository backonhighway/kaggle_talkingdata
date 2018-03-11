import pandas as pd


def add_count(df):
    df["ip_count"] = df.groupby("ip")["channel"].transform('count')
    df["app_count"] = df.groupby("app")["channel"].transform('count')
    df["device_count"] = df.groupby("device")["channel"].transform('count')
    df["os_count"] = df.groupby("os")["channel"].transform('count')
    df["channel_count"] = df.groupby("channel")["ip"].transform('count')
    df["user_count"] = df.groupby(["ip", "device", "os"])["channel"].transform('count')
    df["kernel_qty"] = df.groupby(["ip", "time_day", "time_hour"])["channel"].transform('count')


def add_click_times(df):
    df['time_day'] = df.click_time.str[8:10]
    df['time_hour'] = df.click_time.str[11:13]
    #df['time_min'] = df.click_time.str[14:16]
    #df['time_sec'] = df.click_time.str[17:20]
