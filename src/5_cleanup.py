import os

import dotenv

from utils import execute_cli_command, start_action, end_action

dotenv.load_dotenv('.env')


def main():
    print()
    delete_resource_group()

    delete_app_registration()

    remove_azure_cli_ml_extension()


def delete_resource_group():
    resource_group_name = os.getenv('RESOURCE_GROUP')
    action_text = f'Delete Resource Group "{resource_group_name}"'
    start_action(action_text)

    execute_cli_command(f'az group delete' +
                        f' --name {resource_group_name}' +
                        f' --yes')

    end_action(action_text)


def delete_app_registration():
    app_registration_id = os.getenv('AMLW_CLIENT_ID')
    action_text = 'Delete App Registration'
    start_action(action_text)

    execute_cli_command(f'az ad app delete' +
                        f' --id {app_registration_id}')

    end_action(action_text)


def remove_azure_cli_ml_extension():
    added_extension = os.getenv('AZURE_CLI_ML_EXTENSION_ADDED')
    if added_extension == str(False):
        return

    action_text = 'Removing previously added ML extension of Azure CLI'
    start_action(action_text)

    execute_cli_command('az extension remove' +
                        ' --name ml')

    end_action(action_text)


if __name__ == '__main__':
    main()
