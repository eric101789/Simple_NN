import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('result/train/csv/train_epoch100_logs.csv')
plt.figure(dpi=500)
plt.plot(df['accuracy'], label='training_accuracy')
plt.plot(df['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.legend()
plt.title('Accuracy Epoch=100')
plt.savefig('result/train/csv/acc_epoch100.png')
plt.show()

plt.figure(dpi=500)
plt.plot(df['loss'], label='training_loss')
plt.plot(df['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0.0, 0.2)
plt.legend()
plt.title('Loss Epoch=100')
plt.savefig('result/train/csv/loss_epoch100.png')
plt.show()