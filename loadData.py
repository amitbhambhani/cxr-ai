import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import killWindows

# capital vars are constant
Height = 256
Width = 256

path = "/Users/amit/Downloads/MontgomerySet/"
imagesPath = f"{path}CXR_png/*.png"
leftMaskPath = f"{path}ManualMask/leftMask/*.png"
rightMaskPath = f"{path}ManualMask/rightMask/*.png"


listOfImages = glob.glob(imagesPath)
listOfLeftMaskImages = glob.glob(leftMaskPath)
listOfRightMaskImages = glob.glob(rightMaskPath)

# check correct number of files
# print(len(listOfImages))
# print(len(listOfLeftMaskImages))
# print(len(listOfRightMaskImages))


images = []
masks = []

print("Load training images and masks")

for imageFile, leftFile, rightFile in tqdm(
    zip(listOfImages, listOfLeftMaskImages, listOfRightMaskImages),
    total=len(listOfImages),
):
    image = cv2.imread(imageFile, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (Width, Height))
    image = image / 255.0
    image = image.astype(np.float32)
    images.append(image)

    leftMask = cv2.imread(leftFile, cv2.IMREAD_GRAYSCALE)
    rightMask = cv2.imread(rightFile, cv2.IMREAD_GRAYSCALE)

    mask = leftMask + rightMask
    mask = cv2.resize(mask, (Width, Height))

    for row in mask:
        for i in range(len(row)):
            if row[i] > 0:
                row[i] = 1
    masks.append(mask)

print("Loading completed.")

imagesNP = np.array(images)
masksNP = np.array(masks)
masksNP = masksNP.astype(int)

# iShape = imagesNP.shape
# mShape = masksNP.shape

# print(iShape)
# print(mShape)


# split data to train and validate
split = 0.1  # 90% of images used for training, 10% used for validation

trainImages, validImages = train_test_split(
    imagesNP, test_size=split, random_state=42
)
trainMasks, validMasks = train_test_split(
    masksNP, test_size=split, random_state=42
)

print("Saving data...")

np.save("/Users/amit/Downloads/UNET Data/trainImages.npy", trainImages)
np.save("/Users/amit/Downloads/UNET Data/trainMasks.npy", trainMasks)

np.save("/Users/amit/Downloads/UNET Data/validImages.npy", validImages)
np.save("/Users/amit/Downloads/UNET Data/validMasks.npy", validMasks)

print("Saving completed.")
