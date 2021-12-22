#! /usr/bin/env python3
from import_all import *

pd.set_option('max_columns', None)
warnings.filterwarnings("ignore")


def now():
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


def elapsed(t):
    total = np.round(time.time() - t, 2)
    seconds = np.round(total % 60, 2)
    minutes = int(total // 60 % 60)
    hours = int(total // 60 // 60)
    return str(hours) + ":" + str(minutes) + ":" + str(seconds)


def x_station_by_day(x: pd.DataFrame):
    x = x.drop("hour", axis=1)
    x = x.drop("hour_sin", axis=1)
    x = x.drop("hour_cos", axis=1)

    actions = {col: np.mean for col in x.columns}
    x["Id"] = x["station_id"].astype(str) + "_" + x["day"].astype(str)
    x['Id'] = x['Id'].astype("category")
    set_first = ["station_id", "altitude (m)", "latitude", "longitude", "latitude_idx", "longitude_idx", "month",
                 "month_cos", "month_sin", "day", "shore_distance (m)"]
    for sf in set_first:
        actions[sf] = "first"

    x = x.groupby(["Id"]).agg(actions)

    return x


def save_file(x, save_path, index=False):
    # save in file
    if save_path:
        if not os.path.exists("/".join(save_path.split("/")[:-1])):
            os.mkdir("/".join(save_path.split("/")[:-1]))
        x.to_csv(save_path, index=index)
        print(f"file saved in '{save_path}'")


if __name__ == "__main__":
    t_total = time.time()
    print("start aggregation")

    t_test = time.time()
    print("\nX_all_test :", now())
    df = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_test_imputed.csv")
    df = x_station_by_day(df)
    save_file(df, "../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_test_imputed_by_day.csv")
    del df
    print("DONE - elapsed :", elapsed(t_test))

    t_2016 = time.time()
    print("\nX_all_2016 :", now())
    df = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2016_imputed.csv")
    df = x_station_by_day(df)
    save_file(df, "../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2016_imputed_by_day.csv")
    del df
    print("DONE - elapsed :", elapsed(t_2016))

    t_2017 = time.time()
    print("\nX_all_2016 :", now())
    df = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2017_imputed.csv")
    df = x_station_by_day(df)
    save_file(df, "../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2017_imputed_by_day.csv")
    del df
    print("DONE - elapsed :", elapsed(t_2017))

    print("aggregation complete :", now(), " - total elapsed :", elapsed(t_total))
