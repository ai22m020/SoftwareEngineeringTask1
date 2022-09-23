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

# load image and fit it to match our training data (28x28 px, grayscale, white line on black background)
# convert('L') converts the image into grayscale
img = Image.open(image).convert('L')
myImage = img.resize((28, 28))
# invert colors if image has more whites than blacks (e.g. if it has a black line on white background)
inverted = False
if np.average(np.reshape(np.array(myImage), (28 * 28))) > 128.:
    myImage = ImageOps.invert(myImage)
    inverted = True

# put 2d pixel values into one dimension
shaped = np.reshape(np.array(myImage), (28 * 28))

# calculate brightness mean
avgPixel = np.average(shaped)

# code for showing the preprocessed image
# pyplot.imshow(myImage, cmap=pyplot.get_cmap('gray'))
# pyplot.show()

# output prediction
predictions = loaded_model.predict([shaped])
for index, score in enumerate(predictions[0]):
    if score == 1:
        print('Actual number:', actualValue, ', prediction of knn model with k=', usedK, ' is:', index,
              '. Average pixel darkness (out of 255):', avgPixel, ', Inverted:', inverted)
