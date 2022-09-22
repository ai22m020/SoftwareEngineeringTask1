import pickle
import matplotlib.pyplot as pyplot
from PIL import Image
from PIL import ImageOps
import numpy as np

# load the model from disk
loaded_model = pickle.load(open('model.sav', 'rb'))

actualValue = 4
usedK = 5
image = '4.png'
img = Image.open(image).convert('L') # L ... greyscale
#

myImage = img.resize((28, 28))
inverted = False

if np.average(np.reshape(np.array(myImage), (28 * 28))) > 128.:
    myImage = ImageOps.invert(myImage)
    inverted = True

shaped = np.reshape(np.array(myImage), (28 * 28))

avgPixel = np.average(shaped)


pyplot.imshow(myImage, cmap=pyplot.get_cmap('gray'))
pyplot.show()


predictions = loaded_model.predict([shaped])

for id, score in enumerate(predictions[0]):
    if score == 1:
        print('Actual number:', actualValue,', prediction of knn model with k=', usedK, ' is:', id,'. Average pixel '
                'darkness (out of 255):', avgPixel,', Inverted:', inverted)