import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,adam
from keras import backend as K
from keras.models import model_from_json
import matplotlib.pyplot as plt
from numpy import *
K.set_image_dim_ordering('th')
import os
import theano
from PIL import Image
from numpy import *
K.set_image_dim_ordering('th')
from scipy import misc
from scipy.misc import imread
#for calling the variables defined in my_config python file
from my_config import*
# load json and create model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
#loading testing images
testpath = "dataset\ear_processed\ear_1"
listing = os.listdir(testpath)
test_samples=size(listing)
print(test_samples)
for file in listing:
    #img = Image.open(testpath + '/' + file)
    print("image has been read")
    #plt.imshow(img)
    #plt.show()
    im1 = array(Image.open(testpath + '/'+ listing[0])) # open one image to get size
    m,n = im1.shape[0:2] # get the size of the images
    imnbr = len(listing) # get the number of images
# create matrix to store all flattened images
    immatrix = array([array(Image.open('processed'+ '/' + im2)).flatten()
              for im2 in listing],'f')
print(immatrix.shape)
img = immatrix.reshape(immatrix.shape[0], img_channels, img_rows, img_cols)
print(img.shape)
from sklearn.metrics import classification_report,confusion_matrix
Y_pred = model.predict(img)
print(Y_pred)
y_pred = numpy.argmax(Y_pred, axis=1)
print(y_pred)
p=model.predict_proba(img) # to predict probability





 




