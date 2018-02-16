import httplib2
import os

from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient import discovery

# TODO: Implement initializing spreadsheet

FIELD_NAMES = [
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


def append_row(service, spreadsheet_id, param_dict):
    # The A1 notation of a range to search for a logical table of data.
    # Values will be appended after the last row of the table.
    range_ = 'A1:A{}'.format(len(FIELD_NAMES))
    # How the input data should be interpreted.
    value_input_option = 'USER_ENTERED'
    # How the input data should be inserted.
    insert_data_option = 'INSERT_ROWS'

    value_range_body = {
        "range": range_,
        "majorDimension": 'ROWS',
        "values": [[str(param_dict[field_name]) for field_name in FIELD_NAMES ]]
    }

    request = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_,
        valueInputOption=value_input_option,
        insertDataOption=insert_data_option,
        body=value_range_body)
    response = request.execute()


def get_row(service, spreadsheet_id, param_dict):
    range_ = 'C:C'
    major_dimension = 'COLUMNS'

    request = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_,
        majorDimension=major_dimension)
    response = request.execute()

    try:
        row_idx = response['values'][0].index(param_dict['model_dir'])
        return row_idx + 1
    except ValueError:
        return None


def update_experiment(service, spreadsheet_id, param_dict, start_col, end_col, values):
    row_num = get_row(service, spreadsheet_id, param_dict)
    value_input_option = 'USER_ENTERED'
    range_ = '{1}{0}:{2}{0}'.format(row_num, start_col, end_col)
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
    response = request.execute()

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
