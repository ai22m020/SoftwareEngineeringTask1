import pickle
import matplotlib.pyplot as pyplot
from PIL import Image
from PIL import ImageOps
import numpy as np
import sys


# define image size and greyscale limit
width = 28
height = 28
greyscale_limit = 128


# load the model from disk
loaded_model = pickle.load(open('model.sav', 'rb'))

# read image path from command line argument
if len(sys.argv) > 1:
    image = sys.argv[1]
else:
    image = '4.png'

# open image in greyscale mode ('L')
img = Image.open(image).convert('L')

# resize image to fit training data
myImage = img.resize((width, height))

# calculate average and invert colors if necessary
if np.average(np.reshape(np.array(myImage), (width * height))) > greyscale_limit:
    myImage = ImageOps.invert(myImage)

# reshape image
shaped = np.reshape(np.array(myImage), (width * height))

# plot image
pyplot.imshow(myImage, cmap=pyplot.get_cmap('gray'))
pyplot.show()


print(loaded_model.predict([shaped]))


