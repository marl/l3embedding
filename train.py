import argparse
from l3embedding.train import *


def parse_arguments():
    """
    Parse arguments from the command line


    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Train an L3-like audio-visual correspondence model')

    parser.add_argument('-e',
                        '--num-epochs',
                        dest='num_epochs',
                        action='store',
                        type=int,
                        default=150,
                        help='Maximum number of training epochs')

    parser.add_argument('-es',
                        '--epoch-size',
                        dest='epoch_size',
                        action='store',
                        type=int,
                        default=512,
                        help='Number of training batches per epoch')

    parser.add_argument('-bs',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of training examples per batch')

    parser.add_argument('-vs',
                        '--validation-size',
                        dest='validation_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='Number of trianing examples in the validation set')

    parser.add_argument('-s',
                        '--num-streamers',
                        dest='num_streamers',
                        action='store',
                        type=int,
                        default=32,
                        help='Number of pescador streamers that can be open concurrently')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-a',
                        '--augment',
                        dest='augment',
                        action='store_true',
                        default=False,
                        help='If True, performs data augmentation on audio and images')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='If True, print detailed messages')

    """
    parser.add_argument('train_csv_path',
                        action='store',
                        type=str,
                        help='Path to training csv file')
    """
    parser.add_argument('train_data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where training set files are stored')

    parser.add_argument('validation_data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where validation set files are stored')

    parser.add_argument('model_id',
                        action='store',
                        type=str,
                        help='Identifier for this model')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')


    return vars(parser.parse_args())


if __name__ == '__main__':
    train(**(parse_arguments()))
