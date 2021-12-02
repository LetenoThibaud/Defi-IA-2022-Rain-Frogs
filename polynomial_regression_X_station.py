#! /usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main(x_station_path: str, features: list):
    df = pd.read_csv(x_station_path)
    features = StandardScaler().fit_transform(df[features])
    target = df["ground_truth"].values

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

    regression_poly = PolynomialFeatures(degree=12,
                                         include_bias=True)
    x_poly_train = regression_poly.fit(x_train)

    regression_lin = LinearRegression()
    regression_lin.fit(x_poly_train, y_train)
    


if __name__ == "__main__":
    main(x_station_path="../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_2nn_imputed_by_day.csv",
         features=["wind_speed",
                   "temperature",
                   "dew_point",
                   "humidity",
                   "wind_direction",
                   "precip",
                   "month",
                   "latitude",
                   "longitude",
                   "height_sta"])
