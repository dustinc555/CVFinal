from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt
import glob
from keras import regularizers
import numpy as np


import cv2
import os


X_data = []
files = glob.glob ("test/*.png")
for myFile in files:
    image = cv2.cvtColor(cv2.imread(myFile), cv2.COLOR_BGR2GRAY)
    X_data.append (image)

X_data = np.array(X_data)

images = np.reshape(X_data, (len(X_data), 28, 28, 1))
autoencoder = load_model('sda.h5')

# Test and display
decoded_imgs = autoencoder.predict(images)


n = 9
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
