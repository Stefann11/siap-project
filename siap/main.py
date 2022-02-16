import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.embedding import OpenNetworkEmbedder, EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

test = pd.read_csv('data/out2.csv', sep=',', index_col=False, dtype='unicode')
df = test.iloc[: , 1:]
df = df.drop(['tconst', 'nconst'], axis=1)
df['startYear'] = df['startYear'].astype(np.int16)
df['isAdult'] = df['isAdult'].astype(bool)
def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)
    return df


test2 = Encoder(df)
#X_train, X_test, y_train, y_test = train_test_split(test2, y, test_size=0.33, random_state=42)
#y = df.pop('genres')
y = test2.genres
X = test2.drop('genres', axis=1)
#y = y.str.split(',', expand=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
#X.iloc[X_train]
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=12)
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# #X_train = X_train.reshape(-1,1)
# X_train = X_train.reshape(-1, 1)
# y_train = y_train.reshape(1, -1)
# #y_train = y_train.reshape(1)
# X_test = np.array(X_test)
# X_test = X_test.reshape(-1, 1)
print("X_train: ", X_train)
print("X_train.shape: ",  X_train.shape)
print("y_train: ", y_train)
print("y_train.shape: ", y_train.shape)
testX = np.matrix(X_train)
testY = np.matrix(y_train)
# testY = np.transpose(testY)
# testY = np.transpose(testY)
print("y_train.transpose.shape: ", testY.shape)
# print(test)
# print(type(test))
# y_train = np.array(y_train, dtype=object)
# y_train = y_train.reshape(-1, 1)
# print(y_train)
# print(y_train.shape)


graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
openne_line_params = dict(batch_size=1000, order=3)
embedder = OpenNetworkEmbedder(
    graph_builder,
    'LINE',
    dimension = testY.shape[1],
    aggregation_function = 'add',
    normalize_weights=True,
    param_dict = openne_line_params
)

clf = EmbeddingClassifier(
    embedder,
    RandomForestRegressor(n_estimators=10),
    MLkNN(k=5)
)

#clf.fit(X_train, y_train)
clf.fit(testX, testY)

predictions = clf.predict(X_test)

print(predictions)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
