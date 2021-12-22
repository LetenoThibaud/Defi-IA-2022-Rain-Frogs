#! /usr/bin/env python3
import pandas
import pandas as pd

from import_all import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

pd.options.mode.chained_assignment = None


def knn_imputation(x, k=2, save_path_scores="", file_type="train"):
    # x = x.dropna(subset=['precip'])

    # Imputation data by interpolation

    if "2016" in file_type or "2017" in file_type:
        print("x_train : get timestamp")
        if "2016" in file_type:
            x["year"] = 2016
        elif "2017" in file_type:
            x["year"] = 2017
        x['datetime'] = pd.to_datetime(x[['year', 'month', 'day', "hour"]])
        x["timestamp"] = x["datetime"].values.astype(np.int64) // 10 ** 9
        x = x.drop("year", axis=1)
        x = x.drop("datetime", axis=1)

        print("x_clean : dropna")
        x_clean = x.dropna()
        print("x_fit : standard, transform")
        x_fit = StandardScaler().fit_transform(x_clean[['timestamp', 'latitude', 'longitude']])
    else:  # file_type == "test":
        print("x_clean : dropna")
        x_clean = x.dropna()
        print("x_fit : standard, transform")
        x_fit = StandardScaler().fit_transform(x_clean[['day', "hour", 'latitude', 'longitude']])
        # we spread the day very far one from another so that the k-nn doesn't confuse the days
        x_fit[:, 0] = x_fit[:, 0] * 1000

    print("get labels with missing values.")
    labels = []
    trigger_warning = True
    for col in x.columns:
        total_na = x[col].isna().sum()
        ratio_na = np.round(total_na / len(x), 2)
        if ratio_na >= 0.25:
            if trigger_warning:
                print("WARNING : high na ratio (>0.25) for following feature(s) :")
                trigger_warning = False
            print(col, "-" * (70 - len(col)), ":", total_na, "-" * (10 - len(str(total_na))), "ratio :", ratio_na)
        if total_na > 0:
            labels.append(col)

    scores = dict()
    print("\nstart loop")
    for label in labels:
        print("\tstart imputation :", label)
        y = x_clean[[label]]

        # Evaluation
        x_train, x_test, y_train, y_test = train_test_split(x_fit, y, test_size=0.2, random_state=42)

        neigh = KNeighborsRegressor(3)  # 0.90 wind speed, 0.78 wind orientation, 0.97 temperature
        neigh.fit(x_train, y_train)

        y_pred = neigh.predict(x_test)
        scores[label] = mean_squared_error(y_pred, y_test)

        # Imputation
        # row with the missing label information
        if "2016" in file_type or "2017" in file_type:
            x_missing = x[x[label].isna()][['timestamp', 'latitude', 'longitude']]
        else:  # file_type=="test":
            x_missing = x[x[label].isna()][["day", "hour", "latitude", "longitude"]]
        # Normalization
        x_missing = StandardScaler().fit_transform(x_missing)

        # we train a new model on the X_station clean dataset without split between train and test
        neigh = KNeighborsRegressor(k)
        neigh.fit(x_fit, x_clean[label])

        y_pred = neigh.predict(x_missing)

        fill_na = pd.DataFrame(y_pred, columns=[label])
        index = np.where(pd.isnull(x[[label]]))[0]
        fill_na["index"] = index
        fill_na.set_index("index", inplace=True)
        x[[label]] = x[[label]].fillna(fill_na)
        print('\t-> Successful imputation of the column:', label)

    print("loop done")

    if save_path_scores:
        print("write scores down")
        if not os.path.exists(save_path_scores):
            os.mkdir(save_path_scores)
        # save the score of precision
        save_path_scores = save_path_scores + f'{k}-NN_imputation_scores_{file_type}.txt'
        with open(save_path_scores, 'w') as f:
            f.write(f"List of imputation score (mean_squared_error) with {k}-NN.\n\n")
            max_len_key = max([len(key) for key in list(scores.keys())])
            max_len_value = max([len(str(value)) for value in list(scores.values())])
            f.write("scores = {\n")
            for key, value in scores.items():
                f.write(f"\t'{str(key).ljust(max_len_key + 1)}' : {str(value).rjust(max_len_value)},\n")
            f.write("}")

    return x


if __name__ == "__main__":
    print("Impute X_all_test.csv")
    df_test = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_test.csv")
    knn_imputation(df_test, k=2, save_path_scores="../preprocessed_data_Defi-IA-2022-Rain-Frogs/", file_type="X_all_test")

    print("Impute X_all_2017.csv")
    df_test = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2017.csv")
    knn_imputation(df_test, k=2, save_path_scores="../preprocessed_data_Defi-IA-2022-Rain-Frogs/",
                   file_type="X_all_2017")

    print("Impute X_all_2016.csv")
    df_test = pd.read_csv("../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2016.csv")
    knn_imputation(df_test, k=2, save_path_scores="../preprocessed_data_Defi-IA-2022-Rain-Frogs/",
                   file_type="X_all_2016")
