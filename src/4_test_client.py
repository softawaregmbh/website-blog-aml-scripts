import json
import os
import ssl
import typing
import urllib.error
import urllib.request

from azureml.core import Workspace, Dataset
from dotenv import load_dotenv

from utils import start_action, end_action

load_dotenv('.env')


def main():
    print()
    x_samples, y_samples = fetch_n_samples(n=10)

    pred_samples = predict_test_samples(x_samples)

    pretty_print_results(pred_samples, y_samples)


def fetch_n_samples(n: int) -> [list, list]:
    action_text = f'Fetch first {n} samples from test set'
    start_action(action_text)

    workspace = Workspace(
        os.getenv('SUBSCRIPTION_ID'),
        os.getenv('RESOURCE_GROUP'),
        os.getenv('AML_WORKSPACE_NAME')
    )

    test = Dataset.get_by_name(workspace, name='MNIST Database - Test Partition').to_pandas_dataframe()

    first_n_samples = test.loc[0:(n - 1)]

    x_samples = first_n_samples.loc[:, first_n_samples.columns != 'label']
    y_samples = first_n_samples['label']

    end_action(action_text)
    return x_samples.values.tolist(), y_samples.values.tolist()


def predict_test_samples(x_samples: list) -> typing.Optional[list]:
    action_text = 'Predicting test samples using published endpoint'
    start_action(action_text)

    # bypass the server certificate verification on client side
    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

    data = {"input_data": x_samples}
    body = str.encode(json.dumps(data))

    request = urllib.request.Request(os.getenv('ENDPOINT_URL'), body, headers={
        'Content-Type': 'application/json',
        'Authorization': ('Bearer ' + os.getenv('ENDPOINT_API_KEY')),
        'azureml-model-deployment': os.getenv('ENDPOINT_MODEL_DEPLOYMENT')
    })

    try:
        response = urllib.request.urlopen(request)
        end_action(action_text)
        return json.loads(response.read())
    except urllib.error.HTTPError as error:
        end_action(action_text, state='failure')
        print(f'\nThe request failed with status code: {str(error.code)}')
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        quit(1)


def pretty_print_results(pred_samples, y_samples):
    print('\n' +
          '┌───────┬────────────┐\n' +
          '│ Label │ Prediction │\n' +
          '├───────┼────────────┤')
    if pred_samples is not None:
        for label, prediction in zip(y_samples, pred_samples):
            success_label = '✅' if label == prediction else '❌'
            print(f'│ {label:<5} │ {prediction}          │ {success_label}')
    print('└───────┴────────────┘')


if __name__ == '__main__':
    main()
