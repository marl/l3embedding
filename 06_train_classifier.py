import argparse
from classifier.train import *


def parse_arguments():
    """
    Parse arguments from the command line


    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Generate embedding data for training an urban sound classification model')

    parser.add_argument('-e',
                        '--num-epochs',
                        dest='num_epochs',
                        action='store',
                        type=int,
                        default=150,
                        help='(MLP) Maximum number of training epochs')

    parser.add_argument('-bs',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='(MLP) Number of training examples per batch')

    parser.add_argument('-ns',
                        '--num-streamers',
                        dest='num_streamers',
                        action='store',
                        type=int,
                        help='Number of open pescador streamers to keep open ' \
                             'while sampling features')

    parser.add_argument('-mr',
                        '--mux-rate',
                        dest='mux_rate',
                        action='store',
                        type=int,
                        help='Pescador poisson rate used for sampling')

    parser.add_argument('-tes',
                        '--train-epoch-size',
                        dest='train_epoch_size',
                        action='store',
                        type=int,
                        help='(MLP) Number of training batches per epoch')

    parser.add_argument('-lr',
                        '--learning-rate',
                        dest='learning_rate',
                        action='store',
                        type=float,
                        default=1e-4,
                        help='(MLP) Optimization learning rate')

    parser.add_argument('-wd',
                        '--weight-decay',
                        dest='weight_decay',
                        action='store',
                        type=float,
                        default=1e-5,
                        help='(MLP) L2 regularization penalty factor')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='If True, print detailed messages')

    parser.add_argument('-mt',
                        '--model-type',
                        dest='model_type',
                        action='store',
                        type=str,
                        default='svm',
                        help='Type of model used for training classifier')

    parser.add_argument('features_dir',
                        action='store',
                        type=str,
                        help='Path to directory where feature files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    parser.add_argument('model_id',
                        action='store',
                        type=str,
                        help='Identifier for this model')

    parser.add_argument('fold_num',
                        action='store',
                        type=int,
                        help='Fold ordinal to train/test with')

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_arguments()
    train(**args)
