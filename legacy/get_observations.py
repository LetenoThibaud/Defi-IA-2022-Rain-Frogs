#! /usr/bin/env python3

from import_all import *


def get_observations(x):
    ## shift X
    # get the observation baseline
    obs = x[{"number_sta", "date", "precip"}]
    # obs.set_index('date', inplace=True)

    # if any NaN on the day, then the value is NaN (24 values per day)
    # obs = obs.groupby('number_sta').resample('D')#.agg(pd.Series.sum, min_count = 24)
    obs['date'] = obs['date'].astype('category')
    obs['number_sta'] = obs['number_sta'].astype('category')
    obs['baseline_obs'] = obs.groupby(['number_sta'])['precip'].shift(1)

    obs = obs.sort_values(by=["number_sta", "date"])
    del obs['precip']
    obs = obs.rename(columns={'baseline_obs': 'precip'})
    # obs_new = obs.reset_index()

    return obs
