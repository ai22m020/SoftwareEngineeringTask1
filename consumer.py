import pickle
import matplotlib.pyplot as pyplot
from PIL import Image
from PIL import ImageOps
import numpy as np

# load the model from disk
loaded_model = pickle.load(open('model.sav', 'rb'))

image = '3.png'
img = Image.open(image).convert('L') # L ... greyscale
#

myImage = img.resize((28, 28))


if np.average(np.reshape(np.array(myImage), (28 * 28))) > 128.:
    myImage = ImageOps.invert(myImage)

shaped = np.reshape(np.array(myImage), (28 * 28))

print(np.average(shaped))


pyplot.imshow(myImage, cmap=pyplot.get_cmap('gray'))
pyplot.show()


print(loaded_model.predict([shaped]))


