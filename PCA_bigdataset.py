import numpy as np
import pandas as pd
from sklearn import decomposition
from pprint import pprint

np.random.seed(5)

print("get and merge merge x_all_2016, x_all_2017")
df = pd.concat([pd.read_csv('../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2016_imputed_by_day.csv'),
                pd.read_csv('../preprocessed_data_Defi-IA-2022-Rain-Frogs/X_all_2017_imputed_by_day.csv')], ignore_index=True)

del df["day"]
del df["month"]
del df["station_id"]
del df["timestamp"]

print("create and fir PCA")
pca = decomposition.PCA()
pca.fit(df)

print("get linear combination of hand-crafted features s.t. expressed variance > 0.01")
var_ratio = pca.explained_variance_ratio_
hc_features = pca.transform(df)

linear_comp = pd.DataFrame(columns=df.columns, data=np.abs(np.round(pca.components_, 2)))
linear_comp["explained_variance_ratio"] = np.round(var_ratio, 4)
linear_comp = linear_comp.reindex(columns=['explained_variance_ratio'] + list(linear_comp.columns[:-1]))
# linear_comp.set_index("explained_variance_ratio", inplace=True)
print("Linear Composition of each hand crafter feature : (WARNING : absolute value and rounded 2 dp)")
pprint(linear_comp)

print("weighted sum :")
for col in linear_comp.columns[1:]:
    print(col.rjust(70), ":", np.round(np.sum(linear_comp[col] * linear_comp["explained_variance_ratio"]), 6))
