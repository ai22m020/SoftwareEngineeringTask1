from keras.datasets import mnist
from matplotlib import pyplot
from sklearn import neighbors
from sklearn import metrics
from tensorflow.python.keras.utils.np_utils import to_categorical
import pickle

data = mnist.load_data()
# print(data)
(X_train, y_train), (X_test, y_test) = data

for i in range(2):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()

# reshape(6000, 28 * 28)
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
print(X_train.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model = neighbors.KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
# evaluate the model and update the accuracies list
score = model.score(X_test, y_test)
print("k=%d, accuracy=%.2f%%" % (5, metrics.accuracy_score(y_test, predictions) * 100))
print("k=%d, f1_score=%.2f%%" % (5, metrics.f1_score(y_test, predictions, average="weighted") * 100))

# save model to file
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))
