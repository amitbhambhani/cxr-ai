import keras.layers as tfl
from keras.models import Model


def convBlock(x, numFilters):
    for _ in range(2):
        x = tfl.Conv2D(numFilters, (3, 3), padding="same")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation("relu")(x)

    return x


def buildModel(shape):
    numFilters = [64, 128, 256, 512]
    inputs = tfl.Input((shape))

    skipX = []
    x = inputs

    # encoder
    for f in numFilters:
        x = convBlock(x, f)
        skipX.append(x)
        x = tfl.MaxPool2D((2, 2))(x)

    # bridge
    x = convBlock(x, 1024)
    numFilters.reverse()
    skipX.reverse()

    # decoder
    for i, f in enumerate(numFilters):
        x = tfl.UpSampling2D((2, 2))(x)
        xs = skipX[i]
        x = tfl.Concatenate()([x, xs])
        x = convBlock(x, f)

    # output
    x = tfl.Conv2D(1, (1, 1), padding="same")(x)
    x = tfl.Activation("sigmoid")(x)

    return Model()
