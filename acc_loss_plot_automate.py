import pandas as pd
from matplotlib import pyplot as plt

# Loop through different epoch sizes
for epoch_size in range(100, 1100, 100):
    df = pd.read_csv(f'Simple_NN_1/result/train/csv/train_epoch{epoch_size}_be_logs.csv')
    plt.figure(dpi=500)
    plt.plot(df['accuracy'], label='training_accuracy')
    plt.plot(df['val_accuracy'], label='validation_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.title('Accuracy Epoch=100')
    plt.savefig(f'result/train/csv/acc_epoch{epoch_size}.png')
    plt.show()

    plt.figure(dpi=500)
    plt.plot(df['loss'], label='training_loss')
    plt.plot(df['val_loss'], label='validation_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0.0, 0.2)
    plt.legend()
    plt.title('Loss Epoch=100')
    plt.savefig(f'result/train/csv/loss_epoch{epoch_size}.png')
    plt.show()
