"""
Training Environment:
Python version = 3.7.X
Tensorflow-GPU = 2.6.0
Keras = 2.6.0 (=Tensorflow-GPU)
CUDA version = 12.1
cuDNN version = 8.8.1.3(cuda12)

Building the simple neural network(NN) with Flatten,and Full Connection layers.
Dataset uses CSI amplitudes directly.
Results and models will export to Simple_NN directory.
"""
import csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from matplotlib import pyplot as plt

dataset = pd.read_csv('csi_amplitudes.csv')
X = dataset.iloc[:, 1:53].values
y = dataset.iloc[:, 53].values

# 轉換成NumPy陣列
X = np.array(X)
y = np.array(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)

# Initialize the Neural Network(NN)
NN_model = Sequential()

# Simple NN
NN_model.add(Flatten())
NN_model.add(Dense(units=128, activation='relu'))
NN_model.add(Dense(units=1, activation='sigmoid'))

# Compiling the NN
# NN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
NN_model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

batch_size = 32
epoch_size = 100

# history = NN_model.fit(X_train,
#                        y_train,
#                        epochs=epoch,
#                        batch_size=batch_size,
#                        steps_per_epoch=150,  # MAX <= 10519*0.75//32
#                        validation_data=(X_val, y_val),
#                        validation_steps=16)  # MAX <= 10519*0.05//32=16

# 訓練模型並將結果寫入CSV文件
csvfile = open('result/train/csv/train_epoch100_logs.csv', 'w', newline='')
fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
for epoch in range(epoch_size):
    history = NN_model.fit(X_train,
                           y_train,
                           epochs=1,
                           steps_per_epoch=246,  # MAX <= 10519*0.75//32
                           validation_data=(X_val, y_val),
                           validation_steps=16)  # MAX <= 10519*0.05//32=16

    # 將訓練和驗證損失、精度寫入CSV文件
    writer.writerow({'epoch': epoch + 1,
                     'train_loss': history.history['loss'][0],
                     'train_accuracy': history.history['accuracy'][0],
                     'val_loss': history.history['val_loss'][0],
                     'val_accuracy': history.history['val_accuracy'][0]})

# 關閉CSV文件
csvfile.close()

NN_model.summary()
NN_model.save('model/train_model_epoch100')

# 評估模型
loss, accuracy = NN_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
