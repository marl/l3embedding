import argparse
import os.path
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

    parser.add_argument('-tes',
                        '--train-epoch-size',
                        dest='train_epoch_size',
                        action='store',
                        type=int,
                        default=512,
                        help='Number of training batches per epoch')

    parser.add_argument('-ves',
                        '--validation-epoch-size',
                        dest='validation_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='Number of validation batches per epoch')

    parser.add_argument('-tbs',
                        '--train-batch-size',
                        dest='train_batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of examples per training batch')

    parser.add_argument('-vbs',
                        '--validation-batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of examples per  batch')

    parser.add_argument('-ts',
                        '--train-num-streamers',
                        dest='train_num_streamers',
                        action='store',
                        type=int,
                        default=32,
                        help='Number of training pescador streamers that can be open concurrently')

    parser.add_argument('-vs',
                        '--validation-num-streamers',
                        dest='validation_num_streamers',
                        action='store',
                        type=int,
                        default=32,
                        help='Number of validation pescador streamers that can be open concurrently')

    parser.add_argument('-tmr',
                        '--train-mux-rate',
                        dest='train_mux_rate',
                        action='store',
                        type=float,
                        default=16.0,
                        help='Poisson distribution parameter for determining number of training samples to take from a streamer')

    parser.add_argument('-vmr',
                        '--validation-mux-rate',
                        dest='validation_mux_rate',
                        action='store',
                        type=float,
                        default=16.0,
                        help='Poisson distribution parameter for determining number of validation samples to take from a streamer')

    parser.add_argument('-lr',
                        '--learning-rate',
                        dest='learning_rate',
                        action='store',
                        type=float,
                        default=1e-4,
                        help='Optimization learning rate')

    parser.add_argument('-ci',
                        '--checkpoint-interval',
                        dest='checkpoint_interval',
                        action='store',
                        type=int,
                        default=10,
                        help='The number of epochs between model checkpoints')

    parser.add_argument('-o'
                        '--ontology-path',
                        dest='ontology_path',
                        action='store',
                        type=str,
                        default=os.path.join(os.path.dirname(__file__), 'resources/ontology.json'),
                        help='Path to AudioSet ontology')

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

    parser.add_argument('--gpus',
                        dest='gpus',
                        type=int,
                        default=1,
                        help='Number of gpus used for data parallelism.')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='If True, print detailed messages')

    parser.add_argument('-tmp',
                        '--train-metadata-path',
                        dest='train_metadata_path',
                        action='store',
                        type=str,
                        help='Path to training csv file(s). Accepts a glob string.')

    parser.add_argument('-vmp',
                        '--validation-metadata-path',
                        action='store',
                        type=str,
                        help='Path to validation csv file. Accepts a glob string.')

    parser.add_argument('-tfp',
                        '--train-filter-path',
                        action='store',
                        type=str,
                        help='Path to training csv file(s). Accepts a glob string.')

    parser.add_argument('-vcp',
                        '--validation-csv-path',
                        action='store',
                        type=str,
                        help='Path to validation csv file. Accepts a glob string.')

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
