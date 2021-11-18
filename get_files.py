#! /usr/bin/env python3

from import_all import *


def get_data_train(path):
    ## import data
    df = pd.read_csv(path, parse_dates=['date'], infer_datetime_format=True)

    # sort data
    df = df.sort_values(by=["number_sta", "date"])

    # set number_sta as category
    df["number_sta"] = df["number_sta"].astype("category")

    return df


def get_data_raw(path):
    ## import data
    df = pd.read_csv(path)
    return df
