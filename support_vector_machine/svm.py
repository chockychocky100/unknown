import pandas as pd          
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

pd.options.mode.chained_assignment = None

data = pd.read_csv("apples_and_oranges.csv")

print(data.head())
training_set, test_set = train_test_split(data, test_size = 0.2, random_state = 1)

X_train = training_set.iloc[:,0:2].values
Y_train = training_set.iloc[:,2].values
X_test = test_set.iloc[:,0:2].values
Y_test = test_set.iloc[:,2].values

classifier = SVC(kernel='linear', random_state = 5)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)

classifier = SVC(kernel='linear', random_state = 1)
classifier.fit(X_train,Y_train)
print('Accuracy Metrics')
print(classification_report(Y_test,Y_pred))

print('Confusion Matrix')
print(confusion_matrix(Y_test,Y_pred))

plt.figure(figsize = (7,7))
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('black', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'orange'))(i), label = j)
plt.title('Apples Vs Oranges')
plt.xlabel('Weight In Grams')
plt.ylabel('Size in cm')
plt.legend()
plt.show()
