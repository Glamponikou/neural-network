import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import tensorflow as tf
from tqdm import tqdm
import random
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical

DATADIR = r"C:\Users\dai16\Desktop\datasets etc\Cyclone_Wildfire_Flood_Earthquake_Database"
CATEGORIES = ["Cyclone", "Earthquake", "Flood","Wildfire"]

for category in CATEGORIES:  # do damaged_nature fires and flood
    path = os.path.join(DATADIR, category)  # create path to categories
    for img in os.listdir(path):  # iterate over each image per categories
        img_array = cv2.imread(os.path.join(path, img))  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        #plt.show()  # display!
        break  # we just want one for now so break
    break  # ...and one more!

print(img_array.shape)
IMG_SIZE = 224


new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')


training_data = []

def create_training_data():
    for category in CATEGORIES:  # do flood fire and damaged_nature

        path = os.path.join(DATADIR,category)  # create path to categories
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1 or a 2).

        for img in tqdm(os.listdir(path)):  # iterate over each image per categories
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR) # convert to array
                img_array= cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
                #img_array=cv2color https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()