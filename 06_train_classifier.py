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

    parser.add_argument('-tbs',
                        '--train-batch-size',
                        dest='train_batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='(MLP) Number of training examples per batch')

    parser.add_argument('-tns',
                        '--train-num-streamers',
                        dest='train_num_streamers',
                        action='store',
                        type=int,
                        help='Number of open pescador streamers to keep open ' \
                             'while sampling features for training data')

    parser.add_argument('-tmr',
                        '--train-mux-rate',
                        dest='train_mux_rate',
                        action='store',
                        type=int,
                        help='Pescador poisson rate used for sampling training data')

    parser.add_argument('-vbs',
                        '--valid-batch-size',
                        dest='valid_batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='(MLP) Number of validation examples per batch')

    parser.add_argument('-vns',
                        '--valid-num-streamers',
                        dest='valid_num_streamers',
                        action='store',
                        type=int,
                        help='Number of open pescador streamers to keep open ' \
                             'while sampling features for validation data')

    parser.add_argument('-vmr',
                        '--valid-mux-rate',
                        dest='valid_mux_rate',
                        action='store',
                        type=int,
                        help='Pescador poisson rate used for sampling validation data')

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

    parser.add_argument('-npf',
                        '--norm-penalty-factor',
                        dest='C',
                        action='store',
                        type=float,
                        default=1e-4,
                        help='(SVM) norm penalization factor')

    parser.add_argument('-sct',
                        '--svm-conv-tolerance',
                        dest='tol',
                        action='store',
                        type=float,
                        default=1e-3,
                        help='(SVM) convergence tolerance threshold')

    parser.add_argument('-smi',
                        '--svm-max-iterations',
                        dest='max_iterations',
                        action='store',
                        type=int,
                        default=1000000,
                        help='(SVM) maximum iterations')

    parser.add_argument('-srpt',
                        '--svm-reg-penalty-type',
                        dest='reg_penalty',
                        action='store',
                        type=str,
                        default='l2',
                        choices=['l1', 'l2', 'elasticnet', 'none'],
                        help='(SVM) maximum iterations')

    parser.add_argument('-gsid',
                        '--gsheet-id',
                        dest='gsheet_id',
                        type=str,
                        help='Google Spreadsheet ID for centralized logging of experiments')

    parser.add_argument('-gdan',
                        '--google-dev-app-name',
                        dest='google_dev_app_name',
                        type=str,
                        help='Google Developer Application Name for using API')

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
