import cv2
import numpy as np
import pandas as pd
import os
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras as K
import tensorflow.keras.backend as Kback
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.fftpack import dct
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras import backend
from math import pi
from math import cos
from math import floor

#train_datagen = K.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split = 1.0)   
train_datagen = K.preprocessing.image.ImageDataGenerator(rescale = 1./255)   

# train_dataset  = train_datagen.flow_from_directory(directory = '/kaggle/input/watermeter-data-recognition/MR-AMR Dataset',
#                                                    target_size = (160,160),
#                                                    class_mode = 'categorical',
#                                                    subset = 'training',
#                                                    shuffle=True,
#                                                    batch_size = 64)

validation_dataset  = train_datagen.flow_from_directory(directory = '/kaggle/input/benchmark/challenges',
                                                   target_size = (160,160),
                                                   class_mode = 'categorical',
                                                   shuffle=False,
                                                   batch_size = 64)

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = Kback.sum(Kback.round(Kback.clip(y_true * y_pred, 0, 1)))
    possible_positives = Kback.sum(Kback.round(Kback.clip(y_true, 0, 1)))
    predicted_positives = Kback.sum(Kback.round(Kback.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + Kback.epsilon())
    recall = true_positives / (possible_positives + Kback.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+Kback.epsilon())
    return f1_val

def voting_mean(Y_pred_1,Y_pred_2):
    Y_pred = 0.6*Y_pred_1+0.4*Y_pred_2
    Y_pred = np.log(Y_pred)
    y_pred = np.argmax(Y_pred, axis=1)
    y_pred = y_pred.astype('int')
    return y_pred

model0 = K.models.load_model("/kaggle/input/snapshots/snapshot_model_1.h5", custom_objects={"f1_score": f1_score})
model1 = K.models.load_model("/kaggle/input/snapshots/snapshot_model_2.h5", custom_objects={"f1_score": f1_score})

y_label = np.asarray(validation_dataset.classes)
y_label = y_label.astype('int')

Y_pred_0 = model0.predict_generator(validation_dataset, 1157)
y_pred_0 = np.argmax(Y_pred_0, axis=1)
y_pred_0 = y_pred_0.astype('int')

Y_pred_1 = model1.predict_generator(validation_dataset, 1157)
y_pred_1 = np.argmax(Y_pred_1, axis=1)
y_pred_1 = y_pred_1.astype('int')

y_pred = voting_mean(Y_pred_0, Y_pred_1)

#Making the Confusion Matrix
cm = confusion_matrix(y_label, y_pred)
disp = ConfusionMatrixDisplay(cm,display_labels=['0','1','2','3','4','5','6','7','8','9'])

#Accuracy
from sklearn.metrics import accuracy_score
print("Testing accuracy:")
print(accuracy_score(y_label, y_pred))
#F1_score
from sklearn.metrics import f1_score
print("Testing F1-score")
print(f1_score(y_label, y_pred, average = 'macro'))
#Precision
from sklearn.metrics import precision_score
print("Testing Precision:")
print(precision_score(y_label, y_pred, average = 'macro'))
#Recall
from sklearn.metrics import recall_score
print("Testing Recall:")
print(recall_score(y_label, y_pred, average = 'macro'))

disp.plot()