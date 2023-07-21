import numpy as np
import pandas as pd
from keras import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Import Dataset
dataset = pd.read_csv('csi_amplitudes.csv')
X = dataset.iloc[:, 1:53].values
y = dataset.iloc[:, 53].values

# 轉換成NumPy陣列
X = np.array(X)
y = np.array(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)

# 放大測試集
X_test_enlarged = []
enlargement_factors = [1.47, 1.24, 1.09, 1.06, 1.00, 0.94, 0.91, 0.76, 0.53]

for factor in enlargement_factors:
    enlarged_images = X_test * factor
    X_test_enlarged.append(enlarged_images)

# 轉換成NumPy陣列
X_test_enlarged = np.array(X_test_enlarged)

# 合併放大後的測試集
X_test_combined = np.concatenate(X_test_enlarged, axis=0)
y_test_combined = np.tile(y_test, len(enlargement_factors))

# Initialising the CNN
classifier = Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Define Learning Rate
epoch_size = 100
initial_learning_rate = 0.001
decay_steps = epoch_size // 4
decay_rate = 0.96
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compiling the CNN
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
history = classifier.fit(X_train,
                         y_train,
                         epochs=epoch_size,
                         batch_size=batch_size,
                         steps_per_epoch=7875 // batch_size,
                         validation_data=(X_val, y_val),
                         validation_steps=525 // batch_size)

# 在模型中測試合併後的測試集
loss, accuracy = classifier.evaluate(X_test_combined, y_test_combined)
print(f'Test Accuracy on Combined Enlarged Test Set:{accuracy:.4f}')

# 儲存模型
classifier.save('model/train_model_epoch100_be_enlarge')
