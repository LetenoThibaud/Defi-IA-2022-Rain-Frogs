#! /usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pprint import pprint


def linear_reg(x_station_path: str, features: list):
    df = pd.read_csv(x_station_path)
    features = StandardScaler().fit_transform(df[features])
    target = df["ground_truth"]  # .values.reshape([1,-1])

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.25,
                                                        random_state=42)

    regression_lin = LinearRegression()
    regression_lin.fit(x_train, y_train)

    score = regression_lin.score(x_test, y_test, )
    print(score)


def poly(x_station_path: str, features: list):
    """
    source : https://www.section.io/engineering-education/polynomial-regression-in-python/

    :param x_station_path:
    :param features:
    :return:
    """
    df = pd.read_csv(x_station_path)
    df = df[features + ["ground_truth"]]
    x = df.iloc[:, 0:-1].values  # extracts features from the dataset
    y = df.iloc[:, -1].values  # extracts the labels from the dataset

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)

    df = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_test_coord_2nn_imputed_by_day.csv")
    df_test = df[features]
    x_pred = df_test.values

    poly_regr = PolynomialFeatures(degree=4)  # our polynomial model is of order
    x_poly_train = poly_regr.fit_transform(x_train)  # transforms the features to the polynomial form
    lin_reg = LinearRegression()  # creates a linear regression object
    lin_reg.fit(x_poly_train, y_train)  # fits the linear regression object to the polynomial features

    score = lin_reg.score(poly_regr.fit_transform(x_test), y_test)
    pprint(score)

    df["Prediction"] = lin_reg.predict(poly_regr.fit_transform(x_pred))

    df_pred = df[["Id", "Prediction"]]
    df_pred.to_csv("./prediction_poly_deg_4.csv")

if __name__ == "__main__":
    # linear_reg(x_station_path="../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_2nn_imputed_by_day.csv",
    #            features=["wind_speed",
    #                      "temperature",
    #                      "dew_point",
    #                      "humidity",
    #                      "wind_direction",
    #                      "precip",
    #                      "month",
    #                      "latitude",
    #                      "longitude",
    #                      "height_sta"])

    poly(x_station_path="../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_2nn_imputed_by_day.csv",
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
