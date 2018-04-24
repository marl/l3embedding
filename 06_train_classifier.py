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

    parser.add_argument('-ps',
                        '--parameter-search',
                        dest='parameter_search',
                        action='store_true',
                        help='If True, parameter search will be run')

    parser.add_argument('-psnv',
                        '--parameter-search-no-valid-fold',
                        dest='parameter_search_valid_fold',
                        action='store_false',
                        help='If True, include validation set in train set and instead get the validation set as a ratio of the training set')

    parser.add_argument('-psvr',
                        '--parameter-search-valid-ratio',
                        dest='parameter_search_valid_ratio',
                        action='store',
                        type=float,
                        default=0.15,
                        help='If no validation fold is used, the ratio of the extended training set to set aside for validation')

    parser.add_argument('-pstwv',
                        '--parameter-search-train-with-valid',
                        dest='parameter_search_train_with_valid',
                        action='store_true',
                        help='If True, retrain with validation set')

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
                        default=1.0,
                        help='(SVM) norm penalization factor')

    parser.add_argument('-sct',
                        '--svm-conv-tolerance',
                        dest='tol',
                        action='store',
                        type=float,
                        default=0.00001,
                        help='(SVM) convergence tolerance threshold')

    parser.add_argument('-smi',
                        '--svm-max-iterations',
                        dest='max_iterations',
                        action='store',
                        type=int,
                        default=-1,
                        help='(SVM) maximum iterations')

    parser.add_argument('-skt',
                        '--svm-kernel-type',
                        dest='kernel',
                        action='store',
                        type=str,
                        default='rbf',
                        choices=['rbf', 'sigmoid', 'linear', 'poly'],
                        help='(SVM) kernel type')

    parser.add_argument('-rfne',
                        '--rf-num-estimators',
                        dest='n_estimators',
                        action='store',
                        type=int,
                        default=100,
                        help='(RF) Number of decision trees in the random forest')

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

    parser.add_argument('-fm',
                        '--feature-mode',
                        dest='feature_mode',
                        action='store',
                        type=str,
                        default='framewise',
                        choices=['framewise', 'stats'],
                        help='Type of inputs used for model')

    parser.add_argument('-mt',
                        '--model-type',
                        dest='model_type',
                        action='store',
                        type=str,
                        default='svm',
                        choices=['svm', 'mlp', 'rf'],
                        help='Type of model used for training classifier')

    parser.add_argument('-no',
                        '--non-overlap',
                        dest='non_overlap',
                        action='store_true',
                        default=False)

    parser.add_argument('-nocs',
                        '--non-overlap-chunk-size',
                        dest='non_overlap_chunk_size',
                        action='store',
                        default=10)

    parser.add_argument('-umm',
                        '--use-min-max',
                        dest='use_min_max',
                        action='store_true',
                        default=False)

    parser.add_argument('features_dir',
                        action='store',
                        type=str,
                        help='Path to directory where feature files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    parser.add_argument('fold_num',
                        action='store',
                        type=int,
                        help='Fold ordinal to train/test with')

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_arguments()
    train(**args)
