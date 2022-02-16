import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.embedding import OpenNetworkEmbedder, EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import tensorflow as tf

test = pd.read_csv('data/out.csv', sep=',', index_col=False, dtype='unicode')
df = test.iloc[: , 1:]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y = df.pop('genres')
X = df

X_train, X_test, y_train, y_test = train_test_split(X.index, y.index, test_size=0.2, random_state=12)
#X.iloc[X_train]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=12)
X_train = np.array(X_train)
y_train = np.array(y_train)
#X_train = X_train.reshape(-1,1)
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 1)

graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
openne_line_params = dict(batch_size=1000, order=3)
embedder = OpenNetworkEmbedder(
    graph_builder,
    'LINE',
    dimension = 5*y_train.shape[1],
    aggregation_function = 'add',
    normalize_weights=True,
    param_dict = openne_line_params
)

clf = EmbeddingClassifier(
    embedder,
    RandomForestRegressor(n_estimators=10),
    MLkNN(k=5)
)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(predictions)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
