import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,adam
from keras import backend as K
from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
K.set_image_dim_ordering('th')

# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 128, 128

# number of channels
img_channels = 1
epochs=6

#%%
#  data

path1 ='/home/ubuntu/Downloads/biometric_cnn/raw1'    #path of folder of images    
path2 ='/home/ubuntu/Downloads/biometric_cnn/processed1'  #path of folder to save images 
listing = os.listdir(path1)
num_samples=size(listing)
print num_samples
for file in listing:
    im = Image.open(path1 + '/' + file)  
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(path2 +'/' +  file, "JPEG")

imlist = os.listdir(path2)
im1 = array(Image.open('processed1' + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('processed1'+ '/' + im2)).flatten()
              for im2 in imlist],'f')
label=numpy.ones((num_samples,),dtype = int)
label[0:101]=0
label[102:201]=1
label[202:]=2

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

(X, y) = (train_data[0],train_data[1])
# number of output classes
nb_classes = 3
# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
mean1 = numpy.mean(X_train) # for finding the mean for centering  to zero
X_train -= mean1
X_test -= mean1
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
def larger_model():
	# create model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(1,img_rows,img_cols)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

##    model.add(ZeroPadding2D((1,1)))
##    model.add(Convolution2D(512, 3, 3, activation='relu'))
##    model.add(ZeroPadding2D((1,1)))
##    model.add(Convolution2D(512, 3, 3, activation='relu'))
##    model.add(ZeroPadding2D((1,1)))
##    model.add(Convolution2D(512, 3, 3, activation='relu'))
##    model.add(MaxPooling2D((2,2), strides=(2,2)))
##
##    model.add(ZeroPadding2D((1,1)))
##    model.add(Convolution2D(512, 3, 3, activation='relu'))
##    model.add(ZeroPadding2D((1,1)))
##    model.add(Convolution2D(512, 3, 3, activation='relu'))
##    model.add(ZeroPadding2D((1,1)))
##    model.add(Convolution2D(512, 3, 3, activation='relu'))
##    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    learning_rate=0.1
    decay_rate=learning_rate/epochs
    momentum=0.9
    sgd=SGD(lr=learning_rate,momentum=momentum,decay=decay_rate,nesterov=False)
	
# Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model
# build the model
model = larger_model()
y_train = y_train.reshape((-1, 1))
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=40, nb_epoch=6, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)

# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = numpy.argmax(Y_pred, axis=1)
print(y_pred)
p=model.predict_proba(X_test) # to predict probability
target_names = ['class 0(ear)', 'class 1(fkp)','class 2(iris)']
#print(classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5 and saving the weights
model.save_weights("model.h5")
print("Saved model to disk")
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)






