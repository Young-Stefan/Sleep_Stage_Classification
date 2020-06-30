# File: Untrained CNN
# Description: This program allows the user to train and save a new CNN.
# Institution: University of Texas at Austin, Department of Biomedical Engineering
# Developer: Shao-Po (Shawn) Huang
# Team Members: Bryce Carr, Ajay Gadwal, Ethan Muyskens, Christian Schonhoeft

# Date Last Modified: 05/11/20

# This program uses keras and joblib.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
import joblib

# Helpful resource for building convolutional neural network:
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

# Documentation for keras:
# https://keras.io/

# Source for CNN design:
# Yildirim O, Baloglu UB, Acharya UR. A Deep Learning Model for Automated Sleep Stages Classification Using PSG Signals.
# Int J Environ Res Public Health. 2019;16(4):599. Published 2019 Feb 19. doi:10.3390/ijerph16040599

# Build the CNN
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, strides=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=13, strides=1, activation='relu'))
model.add(Conv1D(filters=256, kernel_size=7, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(filters=256, kernel_size=7, strides=1, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=4, strides=1, activation='relu'))      
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=6, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(filters=8, kernel_size=5, strides=1, activation='relu'))
model.add(Conv1D(filters=8, kernel_size=2, strides=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the CNN
model.fit(trainX, trainy, epochs=100, verbose=2, batch_size=32)

# Save trained CNN by providing destination
joblib.dump(model,PATH_CNN)
