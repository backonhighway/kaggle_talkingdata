def get_predict_col():
    # click_time, is_attributed, not a predict col
    short_col = [
        'app', 'device', 'os', 'channel',
        'hour', 'idoa_is_last_try',
        'group_i_count', "group_i_hourly_count", "group_i_hourly_count_share",
        "group_ido_count", 'group_idoa_count', 'group_ioac_count', 'group_idoac_count',
        #'group_i_prev_click_time', 'group_i_next_click_time',
        'group_ido_prev_click_time', 'group_ido_next_click_time',
        'group_idoa_prev_click_time', 'group_idoa_next_click_time',
        'group_idoac_prev_click_time', 'group_idoac_next_click_time',
        'group_i_nunique_os', 'group_i_nunique_app', 'group_i_nunique_channel',
        'group_i_ct_sum', 'group_i_ct_std', 'group_ido_ct_sum', 'group_ido_ct_std',
        #"ip_ch_mean", #"ip_ch_count",
        #"ip_weighted_ch_mean",
        #"hourly_ip_ch_mean", "hourly_ip_ch_count",
        #"ip_mean", #"ip_count",
        #'group_ict_max', 'group_ict_std', 'group_ict_mean', 'group_ict_sum',
        #'group_idoct_max', 'group_idoct_std', 'group_idoct_mean', 'group_idoct_sum'
    ]
    return short_col