import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# load dataset
train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

# preprocessing stage
X_train_columns = train.columns.drop(['label'])
X_train = train[X_train_columns]
y_train = train['label']

# scaling stage
train_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = train_scaler.transform(X_train)
test_scaler = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = test_scaler.transform(X_test)

# without scaling => 0.93 and after scaling => 0.95
clf = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
print(clf.score(X_train_scaled, y_train))
# print(clf.predict_proba(X_test))
predicted = clf.predict(X_test_scaled)

# X_train_numpy = X_train.to_numpy().reshape(42000, 28, 28)
# y_train_numpy = y_train.to_numpy().reshape(42000, 1)
X_test_numpy = X_test.to_numpy().reshape(28000, 28, 28)
# fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(X_test_numpy[i], cmap='gray')
    plt.title(f'Predicted Digit: {predicted[i]}')
plt.show()





