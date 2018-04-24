import httplib2
import os

from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient import discovery

# TODO: Implement initializing spreadsheet

EMBEDDING_FIELD_NAMES = [
    'username',
    'model_id',
    'model_dir',
    'git_commit',
    'train_data_dir',
    'validation_data_dir',
    'continue_model_dir',
    'model_type',
    'num_epochs',
    'train_epoch_size',
    'validation_epoch_size',
    'train_batch_size',
    'validation_batch_size',
    'random_state',
    'learning_rate',
    'gpus',
    'checkpoint_interval',
    'latest_epoch',
    'latest_train_loss',
    'latest_validation_loss',
    'latest_train_acc',
    'latest_validation_acc',
    'best_train_loss',
    'best_validation_loss',
    'best_train_acc',
    'best_validation_acc'
]

CLASSIFIER_FIELD_NAMES = [
    'username',
    'model_id',
    'model_dir',
    'git_commit',
    'features_dir',
    'fold_num',
    'parameter_search',
    'parameter_search_valid_fold',
    'parameter_search_valid_ratio',
    'parameter_search_train_with_valid',
    'model_type',
    'feature_mode',
    'train_batch_size',
    'non_overlap',
    'random_state',
    'num_epochs',
    'learning_rate',
    'weight_decay',
    'C',
    'tol',
    'max_iterations',
    'train_loss',
    'valid_loss',
    'train_acc',
    'valid_acc',
    'train_avg_class_acc',
    'valid_avg_class_acc',
    'train_class_acc',
    'valid_class_acc',
    'test_acc',
    'test_avg_class_acc',
    'test_class_acc',
]


# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/sheets.googleapis.com-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'


def get_credentials(application_name, client_secret_file=None, flags=None):
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        if not client_secret_file:
            raise ValueError('Must provide client secret file if credentials do not exist')
        flow = client.flow_from_clientsecrets(client_secret_file, SCOPES)
        flow.user_agent = application_name
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials


def append_row(service, spreadsheet_id, param_dict, sheet_name):
    if sheet_name == 'embedding':
        field_names = EMBEDDING_FIELD_NAMES
    elif sheet_name == 'classifier':
        field_names = CLASSIFIER_FIELD_NAMES
    else:
        raise ValueError('Unknown spreadsheet sheet name: {}'.format(sheet_name))

    # The A1 notation of a range to search for a logical table of data.
    # Values will be appended after the last row of the table.
    range_ = '{}!A1:A{}'.format(sheet_name, len(field_names))
    # How the input data should be interpreted.
    value_input_option = 'USER_ENTERED'
    # How the input data should be inserted.
    insert_data_option = 'INSERT_ROWS'

    value_range_body = {
        "range": range_,
        "majorDimension": 'ROWS',
        "values": [[str(param_dict[field_name]) for field_name in field_names ]]
    }

    request = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_,
        valueInputOption=value_input_option,
        insertDataOption=insert_data_option,
        body=value_range_body)
    response = request_with_retry(request)


def request_with_retry(request, num_retries=50):
    exc = None
    for _ in range(num_retries):
        try:
            response = request.execute()
            break
        except Exception as e:
            exc = e
            continue
    else:
        raise exc

    return response


def get_row(service, spreadsheet_id, param_dict, sheet_name, id_field='model_dir'):
    range_ = '{}!C:C'.format(sheet_name)
    major_dimension = 'COLUMNS'

    request = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_,
        majorDimension=major_dimension)
    response = request_with_retry(request)

    try:
        row_idx = response['values'][0].index(param_dict[id_field])
        return row_idx + 1
    except ValueError:
        return None


def update_experiment(service, spreadsheet_id, param_dict, start_col, end_col, values, sheet_name):
    row_num = get_row(service, spreadsheet_id, param_dict, sheet_name)
    value_input_option = 'USER_ENTERED'
    range_ = '{0}!{2}{1}:{3}{1}'.format(sheet_name, row_num, start_col, end_col)
    value_range_body = {
        "range": range_,
        "majorDimension": 'ROWS',
        "values": [[str(val) for val in values]]
    }

    request = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_,
        valueInputOption=value_input_option,
        body=value_range_body)
    response = request_with_retry(request)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(parents=[tools.argparser])
    parser.add_argument('application_name', type=str, help='Name of Google Developer Application')
    parser.add_argument('client_secret_file', type=str, help='Path to application client secret file')
    flags = parser.parse_args()

    # TODO: Fix this hack
    application_name = flags.application_name
    client_secret_file = flags.client_secret_file
    del flags.application_name
    del flags.client_secret_file

    get_credentials(application_name, client_secret_file=client_secret_file, flags=flags)
