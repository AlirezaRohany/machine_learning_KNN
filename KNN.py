import tensorflow
import keras
import pandas
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

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

# print("\n", buying)

goal_feature = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# print("\n", x_train, y_train, x_test, y_test)

# making a KNN model
k = 9
knn_model = KNeighborsClassifier(n_neighbors=k)

knn_model.fit(x_train, y_train)
accuracy = knn_model.score(x_test, y_test)
print("\n" + "Model accuracy:", accuracy,"\n" )

# predict the test data and compare with the real values
# and showing the indices of nearest neighbors for each test node and corresponding euclidean distance
predictions = knn_model.predict(x_test)
goal_names = ['acc', 'good', 'unacc', 'vgood']

print( "Comparing predictions with the real values and displaying neighbors of each node and corresponding distance:")
print("\n")
for i in range(len(predictions)):
    print("Data:", x_test[i])
    print("Neighbors:", knn_model.kneighbors([x_test[i]], k, return_distance=True))
    print("Predicted class:", goal_names[predictions[i]], "  Real class:", goal_names[y_test[i]],"\n")
