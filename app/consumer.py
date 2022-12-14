import pickle
from PIL import Image
from PIL import ImageOps
import numpy as np


def predictImage(picture_path):
    # definitions
    width = 28
    height = 28
    greyscale_limit = 128
    usedK = 5


    # load the model from disk
    loaded_model = pickle.load(open('model.sav', 'rb'))


    # load image and fit it to match our training data (28x28 px, grayscale, white line on black background)
    # convert('L') converts the image into grayscale
    img = Image.open(picture_path).convert('L')

    myImage = img.resize((width, height))

    # invert colors if image has more whites than blacks (e.g. if it has a black line on white background)
    inverted = False

    # calculate average and invert colors if necessary
    if np.average(np.reshape(np.array(myImage), (width * height))) > greyscale_limit:
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
    # return 'Prediction of knn model with k=', usedK, ' is:', predictions[0], '. Average pixel '
    #                'darkness (out of 255):', avgPixel, ', Inverted:', inverted)
    return '{"Prediction": ' + str(predictions[0]) + '}'



