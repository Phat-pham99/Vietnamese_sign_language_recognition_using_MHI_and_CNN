import os
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras import models, layers, optimizers
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop 
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

gestures = {'uo': 'uong',
           'do': 'doi',
           'vu': 'vui',
           'gi': 'gian',
           'to': 'toi',
           'no': 'none'
            }

gestures_map = {'uong' : 0,
                'doi': 1,
                'vui': 2,
                'gian': 3,
                'toi': 4,
                'none':5
                }
num_classes = 6

def process_image(path):
    img = Image.open(path)
    img = img.resize((50,50))
    img = np.array(img)
    return img

def process_data(x_data, y_data):
    x_data = np.array(x_data, dtype = 'float32')
    #x_data = np.stack((x_data,)*3, axis=-1)
    x_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data, num_classes = num_classes)
    return x_data, y_data

def walk_file_tree(relative_path):
    x_data = []
    y_data = [] 
    for directory, subdirectories, files in os.walk(relative_path):
        for file in files:
            if not file.startswith('.') and (not file.startswith('C_')):
                path = os.path.join(directory, file)
                gesture_name = gestures[file[0:2]]
                y_data.append(gestures_map[gesture_name])
                x_data.append(process_image(path))   

            else:
                continue

    x_data, y_data = process_data(x_data, y_data)
    return x_data, y_data

class Data(object):
    def __init__(self):
        self.x_data = []
        self.y_data = []

    def get_data(self):
        return self.x_data, self.y_data

relative_path = '/content/drive/MyDrive/Dataset_MHI/'
# # This method processes the data
x_data, y_data = walk_file_tree(relative_path)

# Classes labeled as 1-25; Need to change to 0-24 for to_categorical
#y_data = to_categorical(y_data, num_classes = num_classes)

# Chia train và test (training 80% và testing 20%)
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state=5, stratify=y_data)

# chia training set thành 30% validation và 70% training
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.3, random_state=5, stratify=Y_train)

# Normalize pixels
X_train = X_train/255
X_val = X_val/255
X_val= X_val.reshape(-1,160,160,3)
X_train= X_train.reshape(-1,160,160,3)
X_test= X_test.reshape(-1,160,160,3)

print(f'x_data shape: {x_data.shape}')
print(f'y_data shape: {y_data.shape}')
print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'Y_train shape: {Y_train.shape}')
print(f'y_val shape: {Y_val.shape}')

cb1 =  keras.callbacks.EarlyStopping(monitor='accuracy',mode="min" ,patience=5)
cb2 =  ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00005)
cb_list = [cb2]
#Dựng model
model = Sequential()
# Lớp 1
model.add(Conv2D(filters=16, kernel_size=(19,19), padding='valid', activation='relu', input_shape=(160,160,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 
# Lớp 2
model.add(Conv2D(filters=16,kernel_size=(11,11),padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 

# Lớp 3
model.add(Conv2D(filters=32,kernel_size=(7,7),padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 
# Lớp 4
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 
#Dense
model.add(Flatten())
model.add(Dense(1066, activation='relu'))
#model.add(Dense(1066, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

#optimizer = keras.optimizers.SGD(learning_rate=0.5,momentum=0.9, nesterov=False)
optimizer = RMSprop(learning_rate=0.0002, rho=0.9, epsilon=1e-08)

# Dùng hàm mất mát categorical crossentropy
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Định nghĩa parameters của mô hình
epochs = 20
batch_size = 32
model.summary() # Xuất ra cấu trúc của mô hình

# Huấn luyện model và đánh giá bằng các tập dữ liệu X_val, y_val với các tập train, batch_size và số epoch đã chọn.
final = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val),shuffle=True, callbacks=[cb_list])

# Vẽ đồ thị độ chính xác training và validation
plt.plot(final.history['accuracy'][1:100])
plt.plot(final.history['val_accuracy'][1:100])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
#plt.ylim((0,1.000)) # limit y-axis to 90-100% accuracy
plt.show()

# Vẽ đồ thị mất mát training and validation
plt.plot(final.history['loss'][1:100])
plt.plot(final.history['val_loss'][1:100])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
#plt.ylim((0,0.5))
plt.show()

 Đánh giá mô hình bằng tập test
model.evaluate(X_test, Y_test)

from sklearn.metrics import classification_report, confusion_matrix
results = model.predict(X_test) # Dự đoán nhãn
Y_pred_classes = np.argmax(results, axis = 1) 
Y_true = np.argmax(Y_test,axis = 1)

# Tạo confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# Vẽ confusion matrix
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot()
sns.heatmap(confusion_mtx, annot=True, fmt="d");
labels=['uong','doi','vui','gian','toi','none']
              
ax1.xaxis.set_ticklabels(labels); ax1.yaxis.set_ticklabels(labels);
