import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display
from icecream import ic

def get_month(index_day):
    # index_day = index_day % 365 + 1
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
        array_of_months.append(get_month(int(index)))
    return np.array(array_of_months)

def get_clean_data(path_station_coordinates, path_X_data, dataset_type):
    coords = pd.read_csv(path_station_coordinates)
    if dataset_type == 'test':
        df = pd.read_csv(path_X_data)
    else:
        df = pd.read_csv(path_X_data, parse_dates=['date'], infer_datetime_format=True)

    array = df['Id'].astype(str).to_numpy()
    post_array = [x.split("_") for x in array]
    # print(post_array)
    Id = [x[0] for x in post_array]
    days = [x[1] for x in post_array]
    hour = [x[2] for x in post_array]

    if dataset_type == 'test':
        df['index_day'] = days
        df['hour'] = hour
        df['number_sta'] = Id
    else :
        # Warning add if to do it only on train (features does not exist on test)
        del df['date']
        del df['number_sta']

        df['index_day'] = days
        df['hour'] = hour
        df['month'] = get_months_by_array(days)
        df['number_sta'] = Id

    df['number_sta'] = df['number_sta'].astype(int)
    coords['number_sta'] = coords['number_sta'].astype(int)
    df = df.merge(coords, on=['number_sta'], how='left')
    return df


if __name__ == '__main__':

    # PATHS :
    path_station_coordinates = '.././/Other/stations_coordinates.csv'
    path_X_data = '.././Train/Train/X_station_train.csv'

    df = get_clean_data(path_station_coordinates, path_X_data)
    print("Shape with precip Nan", df.shape)
    df = df[df['precip'].notna()]
    print("Shape without precip Nan", df.shape)


