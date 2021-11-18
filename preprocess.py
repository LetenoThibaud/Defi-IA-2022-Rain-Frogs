#! /usr/bin/env python3

from import_all import *

warnings.filterwarnings("ignore")


def preprocess_x_y_columns(x, y=None, data_type="train"):
    # preprocess x
    # only proceed if date is in columns, i.e. if x_train supposedly
    if "date" in x.columns:
        x = x.drop("date", axis=1)
    # only proceed if number_sta is in columns, i.e. if x_train supposedly
    if "number_sta" not in x.columns:
        x["number_sta"] = x["Id"].apply(lambda id: int(id.split("_")[0]))

    x["day"] = x["Id"].apply(lambda id_day_hour: int(id_day_hour.split("_")[1]))
    if data_type == "train":
        x["hour"] = x["Id"].apply(lambda id_day_hour: int(id_day_hour.split("_")[2]))
        x["month"] = x['day'].apply(lambda d: get_month(d))

    x["Id"] = x["Id"].apply(lambda id: "_".join(id.split("_")[:2]))

    # preprocess y
    if type(y) != type(None):
        # only proceed if date is in columns, i.e. if y_train supposedly
        if "date" in y.columns:
            y = y.drop("date", axis=1)
        return x, y
    else:
        return x


def merge_x_station_coord(x, coords_path):
    coords = pd.read_csv(coords_path)
    x['number_sta'] = x['number_sta'].astype(int)
    coords['number_sta'] = coords['number_sta'].astype(int)
    x = x.merge(coords, on=['number_sta'], how='left')
    return x


def merge_x_y(x, y):
    if "number_sta" in x.columns and "number_sta" in y.columns:
        y = y.drop("number_sta", axis=1)
    # merge x and y
    x = x.merge(y, how="left", on="Id")
    # we should get x and y with the same number of columns, else push warning.
    if len(x) != len(y):
        print(f"Warning len(x) != len(y) as {len(x)} != {len(y)}")

    return x


def get_month(day_index):
    # day_index = day_index % 365 + 1
    months = [
        31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ]

    i = 0
    while day_index > months[i]:
        day_index -= months[i]
        i += 1

    return (i + 1) % 12


def aggregate_x(x):
    x['number_sta'] = x['number_sta'].astype("category")
    x['Id'] = x['Id'].astype("category")
    # x['day'] = x['day'].astype("category")
    # x['month'] = x['month'].astype("category")
    # x['lat'] = x['lat'].astype("category")
    # x['lon'] = x['lon'].astype("category")
    # x['height_sta'] = x['height_sta'].astype("category")

    print(21, "\n\n", x)
    x = x.groupby(["Id","number_sta","day","month","lat","lon","height_sta"]).agg(pd.Series.sum)
    print(22, "\n\n", x)

    return x
