from keras.datasets import mnist
from matplotlib import pyplot
from sklearn import neighbors
from sklearn import metrics
from tensorflow.python.keras.utils.np_utils import to_categorical
import pickle

data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data

# code for showing all images in a range
# for i in range(2):
#    pyplot.subplot(330 + 1 + i)
#    pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
#    pyplot.show()

# reshape data (put 2d pixel values into one dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

k_value = 5
model = neighbors.KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# evaluate the model and update the accuracies list
score = model.score(X_test, y_test)
# code for accuracy output
# print("k=%d, accuracy=%.2f%%" % (k_value, metrics.accuracy_score(y_test, predictions) * 100))
# print("k=%d, f1_score=%.2f%%" % (k_value, metrics.f1_score(y_test, predictions, average="weighted") * 100))

# save model to file
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))
