import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display

def get_month(index_day):
    index_day = index_day % 365 + 1
    dict_month = {
        1 : 31,
        2 : 28,
        3 : 31,
        4 : 30,
        5 : 31,
        6 : 30,
        7 : 31,
        8 : 31,
        9 : 30,
        10 : 31,
        11 : 30,
        12 : 31
    }
    for key in dict_month.keys():
        index_day = index_day - dict_month[key]
        if index_day <= 0:
            return key

def get_months_by_array(array_index_day):
    array_of_months = []
    for index in array_index_day:
        array_of_months.append(get_month(index))
    return np.array(array_of_months)

def get_clean_data(path_station_coordinates, path_X_data):
    coords = pd.read_csv(path_station_coordinates)

    param = 'hu'
    df = pd.read_csv(path_X_data, parse_dates=['date'], infer_datetime_format=True)

    df = df.merge(coords, on=['number_sta'], how='left')

    array = df['Id'].astype(str).to_numpy()
    post_array = [x.split("_") for x in array]
    # print(post_array)
    Id = [x[0] for x in post_array]
    month = [x[1] for x in post_array]
    hour = [x[2] for x in post_array]

    # Warning add if to do it only on train (features does not exist on test)
    del df['date']
    del df['number_sta']

    df['number_sta'] = Id
    df['month'] = month
    df['hour'] = hour

    return df


if __name__ == '__main__':

    # PATHS :
    path_station_coordinates  = '.././Other/Other/stations_coordinates.csv'
    path_X_data = '.././Train/Train/X_station_train.csv'

