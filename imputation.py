#! /usr/bin/env python3
from import_all import *


def coordinate_based_imputation_train(X, n_neighbors=3, remove_precip=True):
    list_index_day = list(set(X['day'].tolist()))
    list_index_hour = list(range(24))
    print(f"loop : {len(list_index_day) * len(list_index_hour)} iterations to be expected.")
    i = 0
    data = pd.DataFrame(columns=X.columns)
    t_total = time.time()
    for day in list_index_day:
        for hour in list_index_hour:
            t = time.time()
            print(f"\n{i}/{len(list_index_day) * len(list_index_hour)} - day : {day}".ljust(10), f"hour : {hour}".ljust(10), sep=" - ")
            if i<3465 :
                print("skip")
                i+=1
                continue
            data_to_append = subset_coordinate_based_imputation_train(X, hour, day, n_neighbors=n_neighbors,
                                                                      remove_precip=remove_precip)
            # print(data_to_append)
            data = data.append(data_to_append)
            print(f"elapsed time : {time.time() - t:.2f}s - total : {time.time() - t_total:.2f}s")
            i += 1
    return data


def columns_full_nan(X):
    for feature in X.columns.tolist():
        if X[feature].isnull().sum() == X.shape[1]:
            return True
    return False


def subset_coordinate_based_imputation_train(dataset, hour, day, n_neighbors=3, remove_precip=True):
    X = dataset.loc[(dataset['hour'] == hour) & (dataset['day'] == day)]
    nb_while = 1
    print("start subset loop")
    i = 0
    while columns_full_nan(X):
        print(i, end="..")
        i+=1
        if int(hour) - nb_while >= 0:
            X = X.append(dataset.loc[(dataset['hour'] == int(int(hour) - nb_while)) & (dataset['day'] == day)],
                         ignore_index=True)
        if int(hour) + nb_while <= 23:
            X = X.append(dataset.loc[(dataset['hour'] == int(int(hour) + nb_while)) & (dataset['day'] == day)],
                         ignore_index=True)
        nb_while += 1

    if remove_precip:
        # Search for the NaN in the 'precip' columns and remove the corresponding rows (7%)
        X = X[X['precip'].notna()]

    print("create 'new_dataframe'")
    new_dataframe = pd.DataFrame()
    new_dataframe['number_sta'] = X['number_sta']
    new_dataframe['month'] = X['month']
    new_dataframe['height_sta'] = X['height_sta']
    new_dataframe['Id'] = X['Id']

    # create a dataframe filled with the features we used to compute the KNN
    knn_dataframe = X[['lat', 'lon', 'hour', 'day']]
    # delete features we don't impute in X to save space
    del X['number_sta']
    del X['month']
    del X['hour']
    del X['day']
    del X['height_sta']
    del X['Id']
    del X['lat']
    del X['lon']

    if remove_precip:
        new_dataframe['precip'] = X['precip']
        del X['precip']

    print(f"for loop 'feature in {X.columns.tolist()}'")
    for feature in X.columns.tolist():
        print(feature,end="..")
        start = timeit.default_timer()
        # print(X.loc[:, [feature]])
        knn_dataframe[feature] = X.loc[:, [feature]]  # X[[feature]]
        # ic(knn_dataframe)
        # print('# of missing values in col:', feature, knn_dataframe.isnull().sum().sum())
        del X[feature]
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors,
                             metric='nan_euclidean', weights='distance')
        temp_data = imputer.fit_transform(knn_dataframe)

        print(feature,type(feature))
        print(temp_data,type(temp_data))
        try:
            new_dataframe[feature] = temp_data[0:, 4]
        except Exception as e:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("\n\n\tERROR\n\n")
            print(exc_type, fname, exc_tb.tb_lineno)
            print("\n\n")

            print(feature,  type(feature))
            print(temp_data,type(temp_data))
            exit(-2)

        stop = timeit.default_timer()
        # print('Running Time ' + feature + ':', stop - start)
        del knn_dataframe[feature]
    new_dataframe['lon'] = knn_dataframe[['lon']]
    new_dataframe['lat'] = knn_dataframe[['lat']]
    new_dataframe['day'] = knn_dataframe[['day']]
    new_dataframe['hour'] = knn_dataframe[['hour']]
    return new_dataframe


def coordinate_based_imputation_test(X, n_neighbors=3, remove_precip=True):
    if remove_precip:
        # Search for the NaN in the 'precip' columns and remove the corresponding rows (7%)
        X = X[X['precip'].notna()]

    new_dataframe = pd.DataFrame()
    new_dataframe['number_sta'] = X['number_sta']
    new_dataframe['month'] = X['month']
    new_dataframe['height_sta'] = X['height_sta']
    new_dataframe['Id'] = X['Id']

    # create a dataframe filled with the features we used to compute the KNN
    knn_dataframe = X[['lat', 'lon', 'hour', 'day']]
    # delete features we don't impute in X to save space
    del X['number_sta']
    del X['month']
    del X['hour']
    del X['day']
    del X['height_sta']
    del X['Id']
    del X['lat']
    del X['lon']

    if remove_precip:
        new_dataframe['precip'] = X['precip']
        del X['precip']

    for feature in X.columns.tolist():
        start = timeit.default_timer()
        knn_dataframe[feature] = X[[feature]]
        print('# of missing values in col:', feature, knn_dataframe.isnull().sum().sum())
        del X[feature]
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors,
                             metric='nan_euclidean', weights='distance')
        temp_data = imputer.fit_transform(knn_dataframe)
        new_dataframe[feature] = temp_data[0:, 4]
        stop = timeit.default_timer()
        print('Running Time ' + feature + ':', stop - start)
        del knn_dataframe[feature]
    new_dataframe['lon'] = knn_dataframe[['lon']]
    new_dataframe['lat'] = knn_dataframe[['lat']]
    new_dataframe['day'] = knn_dataframe[['day']]
    new_dataframe['hour'] = knn_dataframe[['hour']]
    return new_dataframe


def coordinate_based_imputation(X, dataset_type="train", n_neighbors=3, remove_precip=True):
    """
    :param X: array containing NaN values
    :param dataset_type: can 'test' or 'train'
    :param n_neighbors: if strategy kNN, gives the number of neighbors to consider
    :param remove_precip: boolean to remove the rows where there is a NaN in the precip column
    :return:
    """
    if dataset_type == 'train':
        return coordinate_based_imputation_train(X, n_neighbors=n_neighbors, remove_precip=remove_precip)
    elif dataset_type == 'test':
        return coordinate_based_imputation_test(X, n_neighbors=n_neighbors, remove_precip=remove_precip)
    else:
        print("WARNING : wrong argument in function coordinate_based_imputation must be 'train' or 'test'")
        return pd.DataFrame()


def knn_imputation(X, type='knn_coord', n_neighbors=3, fill_value=0, remove_precip=True):
    """
    :param X: array containing NaN values
    :param type: strategy to use to replace the values, can be 'knn', 'knn_coord', 'mean', "median", "most_frequent"
                or "constant"
    :param n_neighbors: if strategy kNN, gives the number of neighbors to consider
    :param fill_value: value used if "constant" is chosen
    :param remove_precip: boolean to remove the rows where there is a NaN in the precip column
    :return:
    """
    if type == "knn_coord":
        return coordinate_based_imputation(X, n_neighbors, )
    elif type == 'knn':
        if remove_precip:
            # Search for the NaN in the 'precip' columns and remove the corresponding rows (7%)
            X = X[X['precip'].notna()]
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors,
                             metric='nan_euclidean', weights='distance')
    else:
        if remove_precip:
            # Search for the NaN in the 'precip' columns and remove the corresponding rows (7%)
            X = X[X['precip'].notna()]
        imputer = SimpleImputer(missing_values=np.nan, strategy=type, fill_value=fill_value)
    return imputer.fit_transform(X)


def fill_Y_train_Nan(path_station_coordinates, path_Y_data, n_neighbors=3):
    coords = pd.read_csv(path_station_coordinates)
    df = pd.read_csv(path_Y_data, parse_dates=['date'], infer_datetime_format=True)
    df = df.merge(coords, on=['number_sta'], how='left')
    """for feature in df .columns.tolist():
        print('# of missing values in col:', feature, df[[feature]].isnull().sum().sum())"""
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors,
                         metric='nan_euclidean', weights='distance')
    return imputer.fit_transform(df)


if __name__ == '__main__':
    ic.configureOutput(includeContext=True)
    path_station_coordinates = '.././/Other/stations_coordinates.csv'

    """path_X_data = '.././Train/Train/X_station_train.csv'
    path_Y_data = '.././Train/Train/Y_train.csv'
    # y = fill_Y_train_Nan(path_station_coordinates, path_Y_data)
    X = get_clean_data(path_station_coordinates, path_X_data)
    X = X.head(1000)
    ic(X.isnull().sum().sum())
    new_X = coordinate_based_imputation(X, 'train', remove_precip=True)
    ic(X)
    ic(new_X.columns.tolist())
    ic(new_X.isnull().sum().sum())"""

    path_X_data = '.././Train/Train/X_station_train.csv'
    X = get_clean_data(path_station_coordinates, path_X_data, 'train')
    ic(X.columns.tolist())
    display(X.shape)
    ic(X.isnull().sum().sum())
    new_X = coordinate_based_imputation(X, 'train', remove_precip=False)
    print(new_X)
    display(new_X.shape)
    ic(new_X.isnull().sum().sum())

    # ic(new_X.isnull().sum().sum())

    """
    Traceback (most recent call last):
  File "/home/thibaud/Documents/M2_MLDM/ML_project/Defi-IA-2022-Rain-Frogs/imputation.py", line 195, in <module>
    new_X = coordinate_based_imputation(X, 'train', remove_precip=False)
  File "/home/thibaud/Documents/M2_MLDM/ML_project/Defi-IA-2022-Rain-Frogs/imputation.py", line 129, in coordinate_based_imputation
    return coordinate_based_imputation_train(X, n_neighbors=n_neighbors, remove_precip=remove_precip)
  File "/home/thibaud/Documents/M2_MLDM/ML_project/Defi-IA-2022-Rain-Frogs/imputation.py", line 20, in coordinate_based_imputation_train
    data_to_append = subset_coordinate_based_imputation_train(subset_data)
  File "/home/thibaud/Documents/M2_MLDM/ML_project/Defi-IA-2022-Rain-Frogs/imputation.py", line 63, in subset_coordinate_based_imputation_train
    new_dataframe[feature] = temp_data[0:, 4]
IndexError: index 4 is out of bounds for axis 1 with size 4 
    """
