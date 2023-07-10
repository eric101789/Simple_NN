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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from matplotlib import pyplot as plt

dataset = pd.read_csv('csi_amplitudes.csv')
X = dataset.iloc[:, 1:53].values
y = dataset.iloc[:, 53].values

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
NN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32

history = NN_model.fit(X_train,
                       y_train,
                       epochs=100,
                       batch_size=batch_size,
                       steps_per_epoch=7875 // batch_size,
                       validation_data=(X_val, y_val),
                       validation_steps=525 // batch_size)

NN_model.save('model/train_model_epoch100')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('result/train/Loss_epoch100.png')
plt.show()

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('result/train/acc_epoch100.png')
plt.show()

# 評估模型
loss, accuracy = NN_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

y_pred = NN_model.predict(X_test, batch_size=batch_size, verbose=1)

# 取得每個圖片的預測結果和對應的機率
predicted_class = np.argmax(y_pred, axis=1)
class_probability = np.max(y_pred, axis=1)

# 將預測結果和機率寫入CSV文件(LSTM)
results_df = pd.DataFrame({'predicted_class': predicted_class, 'class_probability': class_probability})
results_df.to_csv('result/test/test_results_epoch100.csv', index=False)
