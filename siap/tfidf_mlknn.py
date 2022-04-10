#Import dependencies
import warnings

import numpy as np
import pandas as pd
import texthero as hero
from sklearn.metrics import precision_score
from texthero import preprocessing

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
import tensorflow as tf
from scipy import sparse
import sklearn.metrics as metrics
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

#load the data into a pandas DataFrame
df = pd.read_csv('data/out3.csv', sep=',', index_col=False, dtype='unicode')
df = df.iloc[: , 1:]

custom_pipeline = [preprocessing.fillna,
                   #preprocessing.lowercase,
                   preprocessing.remove_whitespace,
                   preprocessing.remove_diacritics
                   #preprocessing.remove_brackets
                  ]

df['primaryTitle_clean'] = hero.clean(df['primaryTitle'], custom_pipeline)
df['primaryTitle_clean'] = [n.replace('{','') for n in df['primaryTitle_clean']]
df['primaryTitle_clean'] = [n.replace('}','') for n in df['primaryTitle_clean']]
df['primaryTitle_clean'] = [n.replace('(','') for n in df['primaryTitle_clean']]
df['primaryTitle_clean'] = [n.replace(')','') for n in df['primaryTitle_clean']]

df['startYear_clean'] = hero.clean(df['startYear'], custom_pipeline)
df['startYear_clean'] = [n.replace('{','') for n in df['startYear_clean']]
df['startYear_clean'] = [n.replace('}','') for n in df['startYear_clean']]
df['startYear_clean'] = [n.replace('(','') for n in df['startYear_clean']]
df['startYear_clean'] = [n.replace(')','') for n in df['startYear_clean']]

# df['genres_clean'] = hero.clean(df['genres'], custom_pipeline)
# df['genres_clean'] = [n.replace('{','') for n in df['genres_clean']]
# df['genres_clean'] = [n.replace('}','') for n in df['genres_clean']]
# df['genres_clean'] = [n.replace('(','') for n in df['genres_clean']]
# df['genres_clean'] = [n.replace(')','') for n in df['genres_clean']]

df['category_clean'] = hero.clean(df['category'], custom_pipeline)
df['category_clean'] = [n.replace('{','') for n in df['category_clean']]
df['category_clean'] = [n.replace('}','') for n in df['category_clean']]
df['category_clean'] = [n.replace('(','') for n in df['category_clean']]
df['category_clean'] = [n.replace(')','') for n in df['category_clean']]

df['primaryName_clean'] = hero.clean(df['primaryName'], custom_pipeline)
df['primaryName_clean'] = [n.replace('{','') for n in df['primaryName_clean']]
df['primaryName_clean'] = [n.replace('}','') for n in df['primaryName_clean']]
df['primaryName_clean'] = [n.replace('(','') for n in df['primaryName_clean']]
df['primaryName_clean'] = [n.replace(')','') for n in df['primaryName_clean']]

df2 = pd.DataFrame()

df2['tfidf_primaryTitle'] = (hero.tfidf(df['primaryTitle_clean']))
df2['tfidf_startYear'] = (hero.tfidf(df['startYear_clean']))
# df2['tfidf_genres'] = (hero.tfidf(df['genres_clean']))
df2['tfidf_category'] = (hero.tfidf(df['category_clean']))
df2['tfidf_primaryName'] = (hero.tfidf(df['primaryName_clean']))
#
#
# y = df2.tfidf_genres
# X = df2.drop('tfidf_genres', axis=1)
#
# print(X.head(1))

genres = df['genres']
for i, x in enumerate(genres):
    x = x.split(',')
    genres[i] = x
genres = mlb.fit_transform(genres)
primaryTitle = df2['tfidf_primaryTitle']
for i, x in enumerate(primaryTitle):
    # x = x.mean()
    # primaryTitle[i] = x
    primaryTitle[i] = sum(x) / len(x)
startYear = df2['tfidf_startYear']
for i, x in enumerate(startYear):
    # x = statistics.fmean(x)
    # startYear[i] = x
    startYear[i] = sum(x) / len(x)
category = df2['tfidf_category']
for i, x in enumerate(category):
    # x = x.mean()
    # category[i] = x
    category[i] = sum(x) / len(x)
primaryName = df2['tfidf_primaryName']
for i, x in enumerate(primaryName):
    # x = x.mean()
    # primaryName[i] = x
    primaryName[i] = sum(x) / len(x)

y = genres

df3 = pd.DataFrame()

df3['tfidf_primaryTitle'] = primaryTitle
df3['tfidf_startYear'] = startYear
df3['tfidf_category'] = category
df3['tfidf_primaryName'] = primaryName
X = df3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
# sparse_df = X_train.astype(pd.SparseDtype("float64",0))
# csr_sparse_matrix = sparse_df.sparse.to_coo().tocsr()
# sparse_df2 = X_test.astype(pd.SparseDtype("float64",0))
# csr_sparse_matrix2 = sparse_df2.sparse.to_coo().tocsr()

# graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
# openne_line_params = dict(batch_size=1000, order=3)
# embedder = OpenNetworkEmbedder(
#     graph_builder,
#     'LINE',
#     dimension = 5,
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

classifier = MLkNN(k=3)

# y_train = np.transpose(y_train)
# y_train = sparse.lil_matrix(y_train, dtype=float)
# y_train = np.transpose(y_train)
# y_test = np.transpose(y_test)
# y_test = sparse.lil_matrix(y_test, dtype=float)
# y_test = np.transpose(y_test)
# csr_sparse_matrix = sparse.lil_matrix(csr_sparse_matrix)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(accuracy)

precisionScore_sklearn_microavg = precision_score(y_test, predictions, average='micro', zero_division=0)
precisionScore_sklearn_macroavg = precision_score(y_test, predictions, average='macro', zero_division=0)
print("Micro average:", precisionScore_sklearn_microavg)
print("Macro average:", precisionScore_sklearn_macroavg)