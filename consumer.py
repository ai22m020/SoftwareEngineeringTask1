import pickle
import matplotlib.pyplot as pyplot
from PIL import Image
import numpy as np

# load the model from disk
loaded_model = pickle.load(open('model.sav', 'rb'))

image = 'images.png'
img = Image.open(image).convert('L') # L ... greyscale

myImage = img.resize((28, 28))


shaped = np.reshape(np.array(myImage), (28 * 28))
pyplot.imshow(myImage, cmap=pyplot.get_cmap('gray'))
pyplot.show()


print(loaded_model.predict([shaped, shaped]))


