"""
Training Environment:
Python version = 3.7.X
Tensorflow-GPU = 2.6.0
Keras = 2.6.0 (=Tensorflow-GPU)
CUDA version = 12.1
cuDNN version = 8.8.1.3(cuda12)

Building the simple neural network(NN) with Flatten,and Full Connection layers.
Dataset uses CSI amplitudes photos(8*8 size).
Results and models will export to Simple_NN_1 directory.
"""

import numpy as np
import pandas as pd
# from keras import Sequential
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_csv('dataset1.csv')

# 讀取圖片並進行資料處理
X = []
y = []
for _, row in df.iterrows():
    img = load_img(row['path'], target_size=(8, 8, 1))
    img_array = img_to_array(img) / 255.0  # 正規化像素值
    X.append(img_array)
    y.append(row['state'])

# 轉換成NumPy陣列
X = np.array(X)
y = np.array(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)

# Initialising the CNN
classifier = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
# Define Learning Rate
epoch_size = 100
initial_learning_rate = 0.001
decay_steps = epoch_size//4
decay_rate = 0.96
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)
# Compiling the CNN
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32

history = classifier.fit(X_train,
                         y_train,
                         epochs=epoch_size,
                         batch_size=batch_size,
                         steps_per_epoch=7875 // batch_size,
                         validation_data=(X_val, y_val),
                         validation_steps=525 // batch_size)

classifier.save('Simple_NN_1/model/train_model_epoch100')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Simple_NN_1/result/train/Loss_epoch100.png')
plt.show()

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Simple_NN_1/result/train/acc_epoch100.png')
plt.show()

# 評估模型
loss, accuracy = classifier.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

y_pred = classifier.predict(X_test, batch_size=batch_size, verbose=1)

# 取得每個圖片的預測結果和對應的機率
predicted_class = np.argmax(y_pred, axis=1)
class_probability = np.max(y_pred, axis=1)

# 將預測結果和機率寫入CSV文件(LSTM)
results_df = pd.DataFrame({'predicted_class': predicted_class, 'class_probability': class_probability})
results_df.to_csv('Simple_NN_1/result/test/test_results_epoch100.csv', index=False)
