from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.embedding import EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skmultilearn.embedding import OpenNetworkEmbedder
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import sklearn.metrics as metrics
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

train_csv = pd.read_csv('data/out3.csv', sep=',', index_col=False, dtype='unicode')
df = train_csv.iloc[: , 1:]
mlb = MultiLabelBinarizer()
genres = df['genres']
for i, x in enumerate(genres):
    x = x.split(',')
    genres[i] = x
genres = mlb.fit_transform(genres)

le = preprocessing.LabelEncoder()
le.fit(df['primaryTitle'])
primaryTitle = le.transform(df['primaryTitle'])
le.fit(df['startYear'])
startYear = le.transform(df['startYear'])
le.fit(df['category'])
category = le.transform(df['category'])
le.fit(df['primaryName'])
primaryName = le.transform(df['primaryName'])
df['primaryTitle'] = primaryTitle
df['startYear'] = startYear
df['category'] = category
df['primaryName'] = primaryName
df = df.drop('genres', axis=1)
X = df
y = genres
# print(y)
# print(y.shape)
# print(type(y))
# print(X)
# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# classifier = MLkNN(k=3)
#
# # train
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)

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
accuracy = metrics.accuracy_score(y_test, predictions)
print(accuracy)
