import tensorflow
import keras
import pandas
import numpy
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

# print("knn is real")

# reading data
data = pandas.read_csv("car.data")
print(data.head())

# this module can convert categorical data into numerical data
label_encoder = preprocessing.LabelEncoder()

# making categorical values to numerical values (returns an array for each)
buying = label_encoder.fit_transform(list(data["buying"]))
maint = label_encoder.fit_transform(list(data["maint"]))
door = label_encoder.fit_transform(list(data["door"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
cls = label_encoder.fit_transform(list(data["class"]))

print("\n", buying)

goal_feature = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

print("\n", x_train, y_train, x_test, y_test)

# making a KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(x_train, y_train)
accuracy = knn_model.score(x_test, y_test)
print(accuracy)
