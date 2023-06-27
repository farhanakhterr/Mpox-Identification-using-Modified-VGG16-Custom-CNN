import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
import cv2
import os

X = np.load('model/X.txt.npy') #load X and Y data
Y = np.load('model/Y.txt.npy')

print(Y)

X = X.astype('float32') #normalize image pixel with float values
X = X/255
    
test = X[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X.shape[0]) #shuffling the images
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

if os.path.exists('model/vgg_model.json'):
    with open('model/vgg_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    json_file.close()    
    classifier.load_weights("model/vgg_model_weights.h5")
    classifier._make_predict_function()       
else:
    vgg = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights="imagenet")
    vgg.trainable = False
    classifier = Sequential()
    classifier.add(vgg)
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1,1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test, y_test))
    classifier.save_weights('model/vgg_model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/vgg_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/vgg_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
