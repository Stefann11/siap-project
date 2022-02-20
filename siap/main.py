import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.embedding import OpenNetworkEmbedder, EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.preprocessing import normalize
import sklearn.metrics as metrics
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

test = pd.read_csv('data/out.csv', sep=',', index_col=False, dtype='unicode')
df = test.iloc[: , 1:]
df = df.drop(['tconst', 'nconst'], axis=1)
#df['genres'].str.split(',', expand=True)
#df['startYear'] = df['startYear'].astype(np.int16)
#df['isAdult'] = df['isAdult'].astype(bool)
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
#print(test2)
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
#print("X_train: ", X_train)
# print("X_train.shape: ",  X_train.shape)
#print("y_train: ", y_train)
# print("y_train.shape: ", y_train.shape)
#testX = np.matrix(X_train)
trainY = np.matrix(y_train)
trainY = np.transpose(trainY)
testY = np.matrix(y_test)
testY = np.transpose(testY)
# testY = np.transpose(testY)
#print("y_train.transpose.shape: ", testY.shape)
#print("X_train.shape: ", testX.shape)
#print("y_train.type: ", type(testY))
#print("X_train.type: ", type(testX))
# trainX = sparse.csr_matrix(X_train)
# trainY = sparse.csr_matrix(trainY)
# testX = sparse.csr_matrix(X_test)
trainX = sparse.lil_matrix(X_train)
trainY = sparse.lil_matrix(trainY)
testX = sparse.lil_matrix(X_test)
testY = sparse.lil_matrix(testY)
trainX = normalize(trainX, axis=1, norm='l1')
trainY = normalize(trainY, axis=1, norm='l1')
testX = normalize(testX, axis=1, norm='l1')
testY = normalize(testY, axis=1, norm='l1')
trainX = sparse.lil_matrix(X_train)
trainY = sparse.lil_matrix(trainY)
testX = sparse.lil_matrix(X_test)
testY = sparse.lil_matrix(testY)
# print("X_train.sparse: ", trainX)
# print("X_train.shape: ", trainX.shape)
# print("X_train.type: ", type(trainX))
# print("y_train.sparse: ", trainY)
# print("y_train.shape: ", trainY.shape)
# print("y_train.type: ", type(trainY))
# print(test)
# print(type(test))
# y_train = np.array(y_train, dtype=object)
# y_train = y_train.reshape(-1, 1)
# print(y_train)
# print(y_train.shape)

# graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
# openne_line_params = dict(batch_size=1000, order=3)
# embedder = OpenNetworkEmbedder(
#     graph_builder,
#     'LINE',
#     dimension = 5 * trainY.shape[1],
#     aggregation_function = 'add',
#     normalize_weights=True,
#     param_dict = openne_line_params
# )
#
# clf = EmbeddingClassifier(
#     embedder,
#     RandomForestRegressor(n_estimators=10),
#     MLkNN(k=5)
# )
#
# #clf.fit(X_train, y_train)
# clf.fit(trainX, trainY)
#
# predictions = clf.predict(testX)
#
# print(predictions)

classifier = MLkNN(k=3)

# train
classifier.fit(trainX, trainY)

# predict
predictions = classifier.predict(testX)
y_pred_csr = sparse.csr_matrix(predictions)
#score = classifier.score(testX, testY)
#loss = metrics.hamming_loss(testY, predictions)
#testY = sparse.csr_matrix(testY)
print(type(testY))
print(type(predictions))
testY = testY.toarray()
predictions = predictions.toarray()
print(type(testY))
print(type(predictions))
accuracy = metrics.accuracy_score(testY, predictions)
print(accuracy)  # 0.148
# print(predictions)
# print(testY)
#print(testY.shape)
#print(predictions.shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
