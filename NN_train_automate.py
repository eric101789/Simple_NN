import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import CSVLogger
import tensorflow as tf

# Import Dataset
dataset = pd.read_csv('csi_amplitudes.csv')
X = dataset.iloc[:, 1:53].values
y = dataset.iloc[:, 53].values

# Transform to NumPy array
X = np.array(X)
y = np.array(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)

batch_size = 32

# Loop through different epoch sizes
for epoch_size in range(100, 1100, 100):
    # Initialize the Neural Network(NN)
    NN_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Define Learning Rate
    initial_learning_rate = 0.001
    decay_steps = epoch_size // 4
    decay_rate = 0.96
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # Compiling the NN
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    NN_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    csv_logger = CSVLogger(f'result/train/csv/train_epoch{epoch_size}_be_logs.csv', append=False)

    NN_model.fit(X_train,
                 y_train,
                 epochs=epoch_size,
                 batch_size=batch_size,
                 steps_per_epoch=len(X_train) // batch_size,
                 validation_data=(X_val, y_val),
                 validation_steps=len(X_val) // batch_size,
                 callbacks=[csv_logger])

    NN_model.summary()
    NN_model.save(f'model/train_model_epoch{epoch_size}_be')

    # 評估模型
    loss, accuracy = NN_model.evaluate(X_test, y_test)
    print(f'Test Loss (Epoch {epoch_size}): {loss:.4f}')
    print(f'Test Accuracy (Epoch {epoch_size}): {accuracy:.4f}')
