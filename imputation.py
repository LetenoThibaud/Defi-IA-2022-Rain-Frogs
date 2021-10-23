from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from dataset_cleaner import get_clean_data
from IPython.core.display import display
import timeit
from icecream import ic


def coordinate_based_imputation(X, n_neighbors=3, remove_precip=True):
    if remove_precip:
        # Search for the NaN in the 'precip' columns and remove the corresponding rows (7%)
        X = X[X['precip'].notna()]

    new_dataframe = pd.DataFrame()
    new_dataframe['number_sta'] = X['number_sta']
    new_dataframe['month'] = X['month']
    new_dataframe['height_sta'] = X['height_sta']
    new_dataframe['Id'] = X['Id']

    # create a dataframe filled with the features we used to compute the KNN
    knn_dataframe = X[['lat', 'lon', 'hour', 'index_day']]
    # delete features we don't impute in X to save space
    del X['number_sta']
    del X['month']
    del X['hour']
    del X['index_day']
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
        ic(knn_dataframe)
        print('# of missing values in col:', feature, knn_dataframe.isnull().sum().sum())
        del X[feature]
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors,
                             metric='nan_euclidean', weights='distance')
        temp_data = imputer.fit_transform(knn_dataframe)
        print(temp_data)
        new_dataframe[feature] = temp_data[0:, 4]
        stop = timeit.default_timer()
        print('Running Time ' + feature + ':', stop - start)
        del knn_dataframe[feature]
    new_dataframe['lon'] = knn_dataframe[['lon']]
    new_dataframe['lat'] = knn_dataframe[['lat']]
    new_dataframe['index_day'] = knn_dataframe[['index_day']]
    new_dataframe['hour'] = knn_dataframe[['hour']]
    return new_dataframe


def knn_imputation(X, type='knn_coord', n_neighbors=3, fill_value= 0, remove_precip=True):
    """
    :param X: array containing NaN values
    :param type: strategy to use to replace the values, can be 'knn', 'knn_coord', 'mean', "median", "most_frequent"
                or "constant"
    :param n_neighbors: if strategy kNN, gives the number of neighbors to consider
    :param fill_value: value used if "constant" is chosen
    :param remove_precip: boolean to remove the rows where there is a NaN in the precip column
    :return:
    """
    if type == "coord":
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


if __name__ == '__main__':
    ic.configureOutput(includeContext=True)
    path_station_coordinates = '.././/Other/stations_coordinates.csv'
    path_X_data = '.././Train/Train/X_station_train.csv'
    X = get_clean_data(path_station_coordinates, path_X_data)
    X = X.head(1000)
    ic(X.isnull().sum().sum())
    new_X = coordinate_based_imputation(X, remove_precip=True)
    ic(X)
    ic(new_X)
    ic(new_X.isnull().sum().sum())
