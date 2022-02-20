import numpy
import sklearn.metrics as metrics
from skmultilearn.dataset import load_dataset
from skmultilearn.embedding import OpenNetworkEmbedder
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.embedding import EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')

print(X_train)
print(X_train.shape)
print(type(X_train))
print('-----------------------------------')
print(y_train)
print(y_train.shape)
print(type(y_train))

# graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
# openne_line_params = dict(batch_size=1000, order=3)
# embedder = OpenNetworkEmbedder(
#     graph_builder,
#     'LINE',
#     dimension = 5*y_train.shape[1],
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
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
# print(predictions)