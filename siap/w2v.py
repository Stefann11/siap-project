import statistics

import numpy as np
import pandas as pd
import nltk
import scipy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
import tensorflow as tf
from scipy import sparse
from sklearn.preprocessing import normalize
import sklearn.metrics as metrics
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.embedding import OpenNetworkEmbedder
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import sklearn.metrics as metrics
from skmultilearn.embedding import EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
mlb = MultiLabelBinarizer()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()
lemmatizer = WordNetLemmatizer()

#set random seed for the session
random.seed(1)
np.random.seed(1)

train_csv = pd.read_csv('data/out3.csv', sep=',', index_col=False, dtype='unicode')
df = train_csv.iloc[: , 1:]


import texthero as hero
from texthero import preprocessing
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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#tokenize and tag the card text
card_docs = [TaggedDocument(doc.split(' '), [i])
             for i, doc in enumerate(df.primaryTitle_clean)]
card_docs.extend([TaggedDocument(doc.split(' '), [i])
             for i, doc in enumerate(df.startYear_clean)])
# card_docs.extend([TaggedDocument(doc.split(' '), [i])
#              for i, doc in enumerate(df.genres_clean)])
card_docs.extend([TaggedDocument(doc.split(' '), [i])
             for i, doc in enumerate(df.category_clean)])
card_docs.extend([TaggedDocument(doc.split(' '), [i])
             for i, doc in enumerate(df.primaryName_clean)])
model = Doc2Vec(vector_size=64, window=2, min_count=1, workers=8, epochs = 20)
#build vocab
model.build_vocab(card_docs)
#train model
model.train(card_docs, total_examples=model.corpus_count
            , epochs=model.epochs)
primaryTitle2vec = [model.infer_vector((df['primaryTitle_clean'][i].split(' ')))
            for i in range(0,len(df['primaryTitle_clean']))]
startYear2vec = [model.infer_vector((df['startYear_clean'][i].split(' ')))
            for i in range(0,len(df['startYear_clean']))]
# genres2vec = [model.infer_vector((df['genres_clean'][i].split(' ')))
#             for i in range(0,len(df['genres_clean']))]
category2vec = [model.infer_vector((df['category_clean'][i].split(' ')))
            for i in range(0,len(df['category_clean']))]
primaryName2vec = [model.infer_vector((df['primaryName_clean'][i].split(' ')))
            for i in range(0,len(df['primaryName_clean']))]
import numpy as np
#Create a list of lists
# print(primaryTitle2vec)
dtv= np.array(primaryTitle2vec).tolist()
dtv2= np.array(startYear2vec).tolist()
# dtv3= np.array(genres2vec).tolist()
dtv4= np.array(category2vec).tolist()
dtv5= np.array(primaryName2vec).tolist()
#set list to dataframe column
df['primaryTitle'] = dtv
df['primaryTitle'] = primaryTitle2vec
df['startYear'] = startYear2vec
df['category'] = category2vec
df['primaryName'] = primaryName2vec
df['startYear'] = dtv2
genres = df['genres']
for i, x in enumerate(genres):
    x = x.split(',')
    genres[i] = x
genres = mlb.fit_transform(genres)
primaryTitle = df['primaryTitle']
for i, x in enumerate(primaryTitle):
    x = x.mean()
    primaryTitle[i] = x
startYear = df['startYear']
for i, x in enumerate(startYear):
    x = statistics.fmean(x)
    startYear[i] = x
category = df['category']
for i, x in enumerate(category):
    x = x.mean()
    category[i] = x
primaryName = df['primaryName']
for i, x in enumerate(primaryName):
    x = x.mean()
    primaryName[i] = x
# df['category'] = dtv4
# df['primaryName'] = dtv5
#
y = genres
X = df.drop(['genres', 'primaryTitle_clean', 'startYear_clean', 'category_clean', 'primaryName_clean'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
sparse_df = X_train.astype(pd.SparseDtype("float64",0))
csr_sparse_matrix = sparse_df.sparse.to_coo().tocsr()
sparse_df2 = X_test.astype(pd.SparseDtype("float64",0))
csr_sparse_matrix2 = sparse_df2.sparse.to_coo().tocsr()
classifier = MLkNN(k=3)

# # train
# classifier.fit(csr_sparse_matrix, y_train)
# predictions = classifier.predict(csr_sparse_matrix2)

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

clf.fit(csr_sparse_matrix, y_train)

predictions = clf.predict(csr_sparse_matrix2)
accuracy = metrics.accuracy_score(y_test, predictions)
print(accuracy)