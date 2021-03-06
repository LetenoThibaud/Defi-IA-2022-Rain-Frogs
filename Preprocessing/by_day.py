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


def df_by_day(x: pd.DataFrame):
    if "Unnamed: 0" in x.columns:
        x = x.drop("Unnamed: 0", axis=1)
    x = x.drop("hour", axis=1)
    x = x.drop("hour_sin", axis=1)
    x = x.drop("hour_cos", axis=1)

    actions = {col: np.mean for col in x.columns}
    # del actions["Id"]
    # x["Id"] = x["station_id"].astype(str) + "_" + x["month"].astype(str) + "_" + x["day"].astype(str)
    x['Id'] = x['Id'].astype("category")
    set_first = ["Id","station_id", "altitude (m)", "latitude", "longitude", "latitude_idx", "longitude_idx", "month",
                 "month_cos", "month_sin", "day", "shore_distance (m)"]
    for sf in set_first:
        actions[sf] = "first"

    x = x.groupby(["Id"]).agg(actions)

    if "Id" not in x.columns:
        x = x.reset_index()
    if "Unnamed: 0" in x.columns:
        x = x.drop("Unnamed: 0", axis=1)

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
    df = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_test_final.zip")

    print("\n\nbefore")
    pprint(df.head(72))
    df = df_by_day(df)
    print("\n\nafter")
    pprint(df.head(72))
    print("\n\n")

    df["wind_direction_cos"] = np.cos(df["wind_direction (deg)"] / 360 * 2 * np.pi)
    df["wind_direction_sin"] = np.sin(df["wind_direction (deg)"] / 360 * 2 * np.pi)
    df.loc[df["wind_speed (m/s)"] == 0, "wind_direction_cos"] = 0

    save_file(df, "../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_test_final_by_day.zip")
    del df
    print("DONE - elapsed :", elapsed(t_test))

    t_2016 = time.time()
    print("\nX_all_2016 :", now())
    df = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2016_final.zip")

    print("\n\nbefore")
    pprint(df.head(72))
    df = df_by_day(df)
    print("\n\nafter")
    pprint(df.head(72))
    print("\n\n")

    df["wind_direction_cos"] = np.cos(df["wind_direction (deg)"] / 360 * 2 * np.pi)
    df["wind_direction_sin"] = np.sin(df["wind_direction (deg)"] / 360 * 2 * np.pi)
    df.loc[df["wind_speed (m/s)"] == 0, "wind_direction_cos"] = 0

    save_file(df, "../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2016_final_by_day.zip")
    del df
    print("DONE - elapsed :", elapsed(t_2016))

    t_2017 = time.time()
    print("\nX_all_2016 :", now())
    df = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2017_final.zip")

    print("\n\nbefore")
    pprint(df.head(72))
    df = df_by_day(df)
    print("\n\nafter")
    pprint(df.head(72))
    print("\n\n")

    df["wind_direction_cos"] = np.cos(df["wind_direction (deg)"] / 360 * 2 * np.pi)
    df["wind_direction_sin"] = np.sin(df["wind_direction (deg)"] / 360 * 2 * np.pi)
    df.loc[df["wind_speed (m/s)"] == 0, "wind_direction_cos"] = 0

    save_file(df, "../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2017_final_by_day.zip")
    del df
    print("DONE - elapsed :", elapsed(t_2017))

    print("aggregation complete :", now(), " - total elapsed :", elapsed(t_total))
