#! /usr/bin/env python3

from import_all import *

warnings.filterwarnings("ignore")


def preprocess_x_y_columns(x, y=None, data_type="train"):
    # preprocess x
    # only proceed if date is in columns, i.e. if x_train supposedly
    # if "date" in x.columns:
    #     x = x.drop("date", axis=1)
    # x["date"] = x["date"].apply(lambda d: d.split(":")[0].split(" ")[0])
    x["date"] = pd.to_datetime(x["date"], format='%Y-%m-%d %H')
    x["hour"] = x["date"].apply(lambda d: d.hour)
    x["month"] = x["date"].apply(lambda d: d.month)
    x["timestamp"] = x["date"].apply(lambda d: d.timestamp())

    # only proceed if number_sta is in columns, i.e. if x_test supposedly
    if "number_sta" not in x.columns:
        x["number_sta"] = x["Id"].apply(lambda i: int(i.split("_")[0]))

    # x["day"] = x["date"].apply(lambda date: date.day)

    # if data_type == "test":
    #     x["day"] = x["Id"].apply(lambda id_day_hour: int(id_day_hour.split("_")[1]))
    #     x["hour"] = x["Id"].apply(lambda id_day_hour: int(id_day_hour.split("_")[2]))
    #     x["month"] = x['day'].apply(lambda d: get_month(d))

    x["Id"] = x["Id"].apply(lambda i: "_".join(i.split("_")[:2]))

    # preprocess y
    if type(y) != type(None):
        # only proceed if date is in columns, i.e. if y_train supposedly
        if "date" in y.columns:
            y["date"] = pd.to_datetime(y["date"], format='%Y-%m-%d')
            y["date"] = y["date"].apply(lambda date: date.day)
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
    if "date" in x.columns and "date" in y.columns:
        y = y.drop("date", axis=1)
    # merge x and y
    x = x.merge(y, how="left", on="Id")
    # we should get x and y with the same number of columns, else push warning.
    # if len(x) != len(y):
    #     print(f"Warning len(x) != len(y) as {len(x)} != {len(y)}")

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


def get_ground_truth(x):
    first_date = datetime.datetime(2016, 1, 1)
    last_date = datetime.datetime(2017, 12, 31)
    x = x.sort_values(by=["number_sta", "date"])
    x['number_sta'] = x['number_sta'].astype('category')
    x = x.sort_values(['number_sta', 'date'])
    # get the observation baseline
    base_obs = x[{"number_sta", "date", "precip"}]
    base_obs.set_index('date', inplace=True)

    # compute the accumulated rainfall per day with nan management
    # if any NaN on the day, then the value is NaN (24 values per day)
    base_obs = base_obs.groupby('number_sta').resample('D').agg(pd.Series.sum)
    base_obs = base_obs.reset_index(['date', 'number_sta'])
    base_obs['number_sta'] = base_obs['number_sta'].astype('category')

    # Select the observations the day before
    base_obs['baseline_obs'] = base_obs.groupby(['number_sta'])['precip'].shift(-1)
    base_obs = base_obs.sort_values(by=["number_sta", "date"])
    del base_obs['precip']
    base_obs = base_obs.rename(columns={'baseline_obs': 'precip'})

    # get the day indexes (to the final Id)
    date = first_date
    dates = [first_date]
    while date <= (last_date - datetime.timedelta(days=1)):
        date += datetime.timedelta(days=1)
        dates.append(date)

    d_dates = pd.DataFrame(dates, columns=['date'])
    d_dates['day_index'] = d_dates.index

    # create the ID column (id_station + month + index value)
    y = pd.merge(base_obs, d_dates, how="right", on=["date"])
    y = y[y['date'] != last_date]
    y['Id'] = y['number_sta'].astype(str) + '_' + \
              y['day_index'].astype(str)

    # final post-processing
    del y['day_index']
    y = y.rename(columns={'precip': 'ground_truth'})

    return y
