#! /usr/bin/env python3

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from datetime import date
# import re
# import numpy as np
import warnings
from icecream import ic
import pandas as pd

warnings.filterwarnings("ignore")
ic.configureOutput(includeContext=True)


def get_data_train(path):
    ## import data
    df = pd.read_csv(path, parse_dates=['date'], infer_datetime_format=True)

    # sort data
    df = df.sort_values(by=["number_sta", "date"])

    # set number_sta as category
    df["number_sta"] = df["number_sta"].astype("category")

    return df

def get_data_test(path):
    ## import data
    df = pd.read_csv(path)
    return df


def get_observations(x, displ=False):
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

    if displ:
        display(obs)

    return obs


x_train = get_data_train(path='../Train/Train/X_station_train.csv')
y_train = get_data_train(path='../Train/Train/Y_train.csv')

x_test = get_data_test(path="../Test/Test/X_station_test.csv")
baseline = get_data_test(path="../Test/Test/Baselines/Baseline_observation_test.csv")

x_test["day"] = x_test["Id"].apply(lambda id : id.split("_")[1])

ic(x_test)

