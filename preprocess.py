#! /usr/bin/env python3
import time

from import_all import *

from get_files import get_data_train, get_data_raw
from before_imputation import preprocess_x_y_columns, get_ground_truth, merge_x_y, merge_x_station_coord, aggregate_x

from imputation import knn_imputation

pd.set_option('max_columns', None)

warnings.filterwarnings("ignore")


def preprocess_x_station(x_path, y_path, verbose=True, sort=True):
    t = time.time()
    # acquire data from file if input is a file path
    if verbose: print("get x from path", end="")
    # differentiate x_train from x_test
    x = get_data_train(path=x_path)
    # if verbose: print(f"- {time.time() - t:.2f}s\nget y from path", end="")
    # # differentiate y_train from y_test
    # y = get_data_train(path=y_path)

    if verbose: print(f"- {time.time() - t:.2f}s\ndrop na in column precip", end="")
    x = x.dropna(subset=['precip'])

    if verbose: print(f"- {time.time() - t:.2f}s\npreprocess columns", end="")
    x = preprocess_x_y_columns(x)

    if verbose: print(f"- {time.time() - t:.2f}s\nmerge x and coordinates", end="")
    x = merge_x_station_coord(x, coords_path='../Other/Other/stations_coordinates.csv')

    if verbose: print(f"- {time.time() - t:.2f}s\nget ground_truth", end="")
    y = get_ground_truth(x)
    if verbose: print(f"- {time.time() - t:.2f}s\nmerge x and y", end="")
    x = merge_x_y(x, y)

    # if verbose: print(f"- {time.time()-t:.2f}s\naggregate x", end="")
    # x = aggregate_x(x)

    # sort
    if sort:
        if verbose: print(f"- {time.time() - t:.2f}s\nsorting by number_sta then day", end="")
        x.sort_values(["number_sta", "date"], inplace=True)

    if verbose: print(f"- {time.time() - t:.2f}s\nrenaming values", end="")
    x.rename(columns={"ff": "wind_speed",
                      "t": "temperature",
                      "td": "dew_point",
                      "hu": "humidity",
                      "dd": "wind_direction",
                      "lat": "latitude",
                      "lon": "longitude"}, inplace=True)

    if verbose: print(f"- {time.time() - t:.2f}s\ndrop na values from ground truth", end="")
    x = x.dropna(subset=['ground_truth'])

    if verbose: print(f"- {time.time() - t:.2f}s")
    return x


def average_imputation(df_path, verbose=True):
    t_total = time.time()
    if verbose: print("loading data")
    df = get_data_raw(df_path)
    for col in df.columns:
        t = time.time()
        if verbose: print(f"current column : {col}".ljust(30), end=" ")
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(value=df[col].mean())
        else:
            print("no nan", end="")
        if verbose: print(f"elapsed time : {time.time() - t:.2f}s - from beginning : {time.time() - t_total:.2f}s")
    print(f"Done - elapsed time {time.time() - t_total:.2f}s")
    return df


def save_file(x, save_path):
    # save in file
    if save_path:
        if not os.path.exists("/".join(save_path.split("/")[:-1])):
            os.mkdir("/".join(save_path.split("/")[:-1]))
        x.to_csv(save_path, index=False)
        print(f"file saved in '{save_path}'")


def main(task):
    if task == "clean_and_add_coords" or task == 0:
        x_train = preprocess_x_station(x_path='../Train/Train/X_station_train.csv',
                                       y_path='../Train/Train/Y_train.csv',
                                       sort=True)

        save_file(x_train, save_path="../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_raw.csv")

    elif task == "impute_with_average" or task == 1:
        x = average_imputation(df_path="../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_raw.csv")

        save_file(x, "../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_mean_imputed.csv")

    elif task == "impute_with_knn" or task == 2:
        print("Start k-nn imputation")
        x = get_data_raw("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_raw.csv")

        k = 2
        x = knn_imputation(x, k=k, save_path_scores="../preprocessed_data_Defi-IA-2022-Rain-Frogs/")

        save_file(x, f"../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_station_coord_{k}nn_imputed.csv")


if __name__ == "__main__":
    main(task=0)
    main(task=2)
