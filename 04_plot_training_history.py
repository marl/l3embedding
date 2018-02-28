import argparse
import os.path
import matplotlib.pyplot as plt
import csv


def generate_plot(csv_path, display=True, output_path=None):
    """
    Generate a plot of the accuracy and loss of training and validation sets

    Args:
        csv_path: path to training history CSV file
        display: if True, display plot in a window
        output_path: if provided, save plot to the path
    """
    epochs = []
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    csv_path = os.path.abspath(csv_path)

    # Get model ID and timestamp for plot title
    dir_parts = os.path.dirname(csv_path).split('/')
    model_id, timestamp = dir_parts[-2], dir_parts[-1]

    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']) + 1)
            train_acc.append(float(row['acc']))
            train_loss.append(float(row['loss']))
            val_acc.append(float(row['val_acc']))
            val_loss.append(float(row['val_loss']))

    # Plot accuracy
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, val_acc)
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Subset Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Subset Loss')

    plt.suptitle('{}/{}'.format(model_id, timestamp))

    if output_path:
        plt.savefig(output_path)

    if display:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the training and validation accuracy and loss for a given training run')
    parser.add_argument('csv_path', type=str, help='path to training csv file')
    parser.add_argument('--hide-display', '-hd', dest='display', action='store_false', help='if set, do not display plot')
    parser.add_argument('--output-path', '-o', dest='output_path', type=str, help='optional output path for plot image')

    args = parser.parse_args()
    generate_plot(args.csv_path, display=args.display, output_path=args.output_path)
