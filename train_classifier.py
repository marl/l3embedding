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

    parser.add_argument('-es',
                        '--epoch-size',
                        dest='epoch_size',
                        action='store',
                        type=int,
                        default=512,
                        help='(MLP) Number of training batches per epoch')

    parser.add_argument('-bs',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='(MLP) Number of training examples per batch')

    parser.add_argument('-vs',
                        '--validation-size',
                        dest='validation_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='(MLP) Number of trianing examples in the validation set')

    parser.add_argument('-lr',
                        '--learning-rate',
                        dest='learning_rate',
                        action='store',
                        type=float,
                        default=1e-4,
                        help='(MLP) Optimization learning rate')

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

    parser.add_argument('metadata_path',
                        action='store',
                        type=str,
                        help='Path to UrbanSound8K metadata file')

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
