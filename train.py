import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
import pickle

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split 

#path of dataset
path = 'Dataset'
#names of all plant disease
labels = ['Normal', 'Monkeypox']
X = []
Y = []
#function to return integer label for given plat disease name
def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name: #return integer ID as label for given plant disease name
            index = i
            break
    return index

print(labels)

#looping all images from all plant disease folders
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j]) #read image
            img = cv2.resize(img, (32,32)) #resize image
            im2arr = np.array(img)
            im2arr = im2arr.reshape(32,32,3) #resize as colur image
            label = getID(name) #get id or label of plant disease
            X.append(im2arr) #add all image pixel to X array
            Y.append(label) #add label to Y array
            print(name+" "+root+"/"+directory[j]+" "+str(label))
        
X = np.asarray(X)
Y = np.asarray(Y)

np.save('model/X.txt',X) #save X and Y data for future user
np.save('model/Y.txt',Y)

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


if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        cnn_model = model_from_json(loaded_model_json)
    cnn_model.load_weights("model/model_weights.h5")
    cnn_model._make_predict_function()       
else:
    cnn_model = Sequential() #define CNN model and its layers
    cnn_model.add(Conv2D(32, (3, 3), padding="same",input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    cnn_model.add(Activation("relu"))
    cnn_model.add(BatchNormalization(axis=3))
    cnn_model.add(MaxPooling2D(pool_size=(3, 3)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Conv2D(64, (3, 3), padding="same"))
    cnn_model.add(Activation("relu"))
    cnn_model.add(BatchNormalization(axis=3))
    cnn_model.add(Conv2D(64, (3, 3), padding="same"))
    cnn_model.add(Activation("relu"))
    cnn_model.add(BatchNormalization(axis=3))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Conv2D(128, (3, 3), padding="same"))
    cnn_model.add(Activation("relu"))
    cnn_model.add(BatchNormalization(axis=3))
    cnn_model.add(Conv2D(128, (3, 3), padding="same"))
    cnn_model.add(Activation("relu"))
    cnn_model.add(BatchNormalization(axis=3))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1024))
    cnn_model.add(Activation("relu"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(y_train.shape[1]))
    cnn_model.add(Activation("softmax"))    
    print(cnn_model.summary())
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compile the model
    hist = cnn_model.fit(X, Y, batch_size=16, epochs=15, shuffle=True, verbose=2, validation_data=(X_test, y_test)) #start traing model
    cnn_model.save_weights('model/model_weights.h5')            
    model_json = cnn_model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    
