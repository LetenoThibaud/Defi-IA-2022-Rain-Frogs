#! /usr/bin/env python3

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from datetime import date
# import re
# import numpy as np
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

def get_data_train(path):
    ## import data
    df = pd.read_csv(path, parse_dates=['date'], infer_datetime_format=True)

    # sort data
    df = df.sort_values(by=["number_sta", "date"])

    # set number_sta as category
    df["number_sta"] = df["number_sta"].astype("category")

    return df

