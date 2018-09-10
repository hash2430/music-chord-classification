import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    loss_valid_file = './export/rnn_only_frame_do_early_stop/learning_curve/loss_valid_file.npy'
    loss_valid = np.load(loss_valid_file)
    plt.plot(loss_valid)
    plt.ylim(0, 3)
    plt.xlim(0, 50)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Validation loss')
    plt.show()

    acc_valid_file = './export/rnn_only_frame_do_early_stop/learning_curve/acc_valid_file.npy'
    acc_valid = np.load(acc_valid_file)
    acc_valid = acc_valid * 100
    plt.plot(acc_valid)
    plt.title('Valication accuracy')
    plt.ylim(0, 60)
    plt.xlim(0, 50)
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.show()