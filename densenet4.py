import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.3,
    horizontal_flip=True,
    zoom_range = 0.2,
                        )


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)



X_norm = tf.keras.applications.densenet.preprocess_input(X)



densenet = tf.keras.applications.DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classes=4
)

model = Sequential()
model.add(densenet)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



datagen.fit(X_norm)
# fits the model on batches with real-time data augmentation:

model.fit(datagen.flow(X_norm, y, batch_size=32),steps_per_epoch=len(X_norm) / 32, epochs=10)

# evaluate the model
scores = model.evaluate(X_norm, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")

model.summary()

