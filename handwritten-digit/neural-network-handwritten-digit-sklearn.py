import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# load dataset
train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

# preprocessing stage
X_train_columns = train.columns.drop(['label'])
X_train = train[X_train_columns]
y_train = train['label']

# # scaling stage
train_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = train_scaler.transform(X_train)
test_scaler = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = test_scaler.transform(X_test)

# convert to numpy array => suitale for sklearn
# X_train_as_np_format = X_train_scaled.values
# y_train__as_np_format = y_train.to_numpy()
# X_test_as_np_format = X_test_scaled.values
X_train_as_np_format = X_train_scaled
y_train__as_np_format = y_train.to_numpy()
X_test_as_np_format = X_test_scaled

# mlp_clf = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
#                         solver='sgd', tol=1e-4, random_state=1,
#                         learning_rate_init=.1, verbose=True)

mlp_clf = MLPClassifier(hidden_layer_sizes=(15, 5,), solver='adam', max_iter=100,
                        activation='relu', random_state=1, verbose=True)

mlp_clf.fit(X_train_as_np_format, y_train__as_np_format)
print(mlp_clf.score(X_train_as_np_format, y_train__as_np_format))
# # print(clf.predict_proba(X_test))
predicted = mlp_clf.predict(X_test_as_np_format)

X_test_numpy = X_test.to_numpy().reshape(28000, 28, 28)
# fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(X_test_numpy[i], cmap='gray')
    plt.title(f'Predicted Digit: {predicted[i]}')
plt.show()
