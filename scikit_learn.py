from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

iris = datasets.load_iris()
x, y = iris.data[:, :2], iris.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=33)
scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain, ytrain)
ypred = knn.predict(xtest)
print("accuracy score knn = ", accuracy_score(ytest, ypred))
print(xtest, ytest)
