#importing libraries
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.embedding import OpenNetworkEmbedder, EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()
nltk.download('stopwords')
nltk.download('wordnet')

train_csv = pd.read_csv('data/out2.csv', sep=',', index_col=False, dtype='unicode')
df = train_csv.iloc[: , 1:]
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

y = df.genres
X = df.drop('genres', axis=1)
X = X.drop('tconst', axis=1)
X = X.drop('nconst', axis=1)
# y = y.str.split(',', expand=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
train_X = []
test_X = []
train_Y = []
# test2 = np.array(X_train)
# test = test2[0]
# test = train_csv['primaryTitle']
# print(test.shape)

# #text pre processing
# for i in range(0, len(test)):
#     review = re.sub('[^a-zA-Z]', ' ', test[i])
#     review = review.lower()
#     review = review.split()
#     review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
#     review = ' '.join(review)
#     train_X.append(review)
# print(train_X)
#
# # text pre processing
# for i in range(0, len(X_test)):
#     review = re.sub('[^a-zA-Z]', ' ', X_test[i])
#     review = review.lower()
#     review = review.split()
#     review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
#     review = ' '.join(review)
#     test_X.append(review)
#
#train_X = [X_train]
#tf idf
train_X = np.array(X_train)
test_X = np.array(X_test)
train_Y = np.array(y_train)
#print(train_X.shape)
X_train2 = train_X.astype('U')
X_test2 = test_X.astype('U')
y_train2 = train_Y.astype('U')
tf_idf = TfidfVectorizer()
#applying tf idf to training data
X_train_tf = tf_idf.fit_transform(X_train2.ravel())
y_train_tf = tf_idf.fit_transform(y_train2.ravel())
#applying tf idf to training data
X_train_tf = tf_idf.transform(X_train2.ravel())
y_train_tf = tf_idf.transform(y_train2.ravel())
print("n_samples: %d, n_features: %d" % X_train_tf.shape)
print("Y n_samples: %d, n_features: %d" % y_train_tf.shape)
#transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(X_test2.ravel())
print("n_samples: %d, n_features: %d" % X_test_tf.shape)

print(tf_idf.get_params())

# naive_bayes_classifier = MultinomialNB()
# naive_bayes_classifier.fit(X_train_tf, train_Y)
# y_pred = naive_bayes_classifier.predict(X_test_tf)
# print(y_pred)

# graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
# openne_line_params = dict(batch_size=1000, order=3)
# embedder = OpenNetworkEmbedder(
#     graph_builder,
#     'LINE',
#     dimension = 4,
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
# clf.fit(X_train_tf, y_train_tf)
#
# predictions = clf.predict(X_test_tf)
#
# print(predictions)