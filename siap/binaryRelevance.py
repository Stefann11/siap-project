from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.embedding import EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import LinearSVR
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skmultilearn.embedding import OpenNetworkEmbedder
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import sklearn.metrics as metrics
from sklearn.metrics import precision_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
sparse_df = X_train.astype(pd.SparseDtype("float64",0))
csr_sparse_matrix = sparse_df.sparse.to_coo().tocsr()
sparse_df2 = X_test.astype(pd.SparseDtype("float64",0))
csr_sparse_matrix2 = sparse_df2.sparse.to_coo().tocsr()

classifier = BinaryRelevance(
    classifier = SVC(),
    require_dense = [False, True]
)
classifier.fit(csr_sparse_matrix, y_train)
predictions = classifier.predict(csr_sparse_matrix2)
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

precisionScore_sklearn_microavg = precision_score(y_test, predictions, average='micro')
precisionScore_sklearn_macroavg = precision_score(y_test, predictions, average='macro')
print("Micro average:", precisionScore_sklearn_microavg)
print("Macro average:", precisionScore_sklearn_macroavg)
