import argparse
from classifier.train import *


def parse_arguments():
    """
    Parse arguments from the command line


    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Train an urban sound classification model')

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

    parser.add_argument('-tes',
                        '--train-epoch-size',
                        dest='train_epoch_size',
                        action='store',
                        type=int,
                        help='(MLP) Number of training batches per epoch')

    parser.add_argument('-ves',
                        '--validation-epoch-size',
                        dest='validation_epoch_size',
                        action='store',
                        type=int,
                        help='(MLP) Number of validation batches per epoch')

    parser.add_argument('-vs',
                        '--validation-split',
                        dest='validation_split',
                        action='store',
                        type=float,
                        default=0.1,
                        help='(MLP) Percentage of training data used for validation')

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

    parser.add_argument('-f',
                        '--features',
                        dest='features',
                        action='store',
                        type=str,
                        default='l3_stack',
                        help='Type of features to be used in training')

    parser.add_argument('-lf',
                        '--label-format',
                        dest='label_format',
                        action='store',
                        type=str,
                        default='int',
                        help='Type of format used for encoding outputs')

    parser.add_argument('-mt',
                        '--model-type',
                        dest='model_type',
                        action='store',
                        type=str,
                        default='svm',
                        help='Type of model used for training classifier')

    parser.add_argument('-lmp',
                        '--l3embedding-model-path',
                        dest='l3embedding_model_path',
                        action='store',
                        type=str,
                        help='Path to L3 embedding model weights file')

    parser.add_argument('-lmt',
                        '--l3embedding-model-type',
                        dest='l3embedding_model_type',
                        action='store',
                        type=str,
                        default='cnn_L3_orig',
                        help='Type of L3 embedding model')

    parser.add_argument('-hs',
                        '--hop-size',
                        dest='hop_size',
                        action='store',
                        type=float,
                        default=0.25,
                        help='Hop size as a fraction of the window')

    parser.add_argument('-nrs',
                        '--num-random-samples',
                        dest='num_random_samples',
                        action='store',
                        type=int,
                        help='Number of random samples for randomized sampling methods')

    parser.add_argument('-nl',
                        '--no-logging',
                        dest='disable_logging',
                        action='store_true',
                        default=False,
                        help='Disables logging if flag enabled')

    parser.add_argument('-lp',
                        '--log-path',
                        dest='log_path',
                        action='store',
                        default=None,
                        help='Path to log file generated by this script. ' \
                             'By default, the path is "./audiosetdl.log".')

    parser.add_argument('metadata_path',
                        action='store',
                        type=str,
                        help='Path to UrbanSound8K metadata file')

    parser.add_argument('dataset_name',
                        action='store',
                        type=str,
                        help='Name of dataset')

    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where training set files are stored')

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
    args = parse_arguments()
    train(**args)
