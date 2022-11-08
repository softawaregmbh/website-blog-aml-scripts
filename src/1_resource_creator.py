import os
import json
import subprocess
import time
from typing import Union
import uuid

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, IdentityConfiguration, ManagedIdentityConfiguration
from azure.identity import DefaultAzureCredential
import dotenv

CONSOLE_COLOR_RESET_CODE = '\x1b[0m'


def execute_cli_command(command: str) -> Union[str, list, dict]:
    command = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, shell=True)
    output, error_output = command.communicate()

    if command.returncode != 0:
        raise RuntimeError(error_output)

    output = output.strip()

    if output.startswith('[') or output.startswith('{'):
        output = output[: -len(CONSOLE_COLOR_RESET_CODE)] if output.endswith(CONSOLE_COLOR_RESET_CODE) else output
        return json.loads(output)
    elif output.startswith('"') and output.endswith('"'):
        return output[1:-1]
    else:
        return output


def start_action(action_text: str):
    print(f'⚪ {action_text}', end='')


def end_action(action_text: str, state: str = 'success'):
    if state == 'success':
        status_symbol = '🟢'
    elif state == 'skipped':
        status_symbol = '🔵'
    elif state == 'failure':
        status_symbol = '🔴'
    else:
        raise ValueError(f'State {state} unhandled.')

    print(f'\r{status_symbol} {action_text}')


def main():
    resource_group_name = f'rg-azure-ml-showcase-{str(uuid.uuid4())[:8]}'
    aml_workspace_name = f'mlw-mlshowcase-{str(uuid.uuid4())[:8]}'
    managed_id_name = 'id-compute-cluster'
    compute_cluster_name = 'cluster-standard-ds3-v2'
    environment_name = 'env-digit-classifier'

    # TODO: Check if azure cli is installed
    # TODO: Check if azure cli ml extension is enabled

    print()
    subscription_id = confirm_used_subscription()
    print()

    create_resource_group(resource_group_name)

    storage_account_id, storage_account_name = create_azure_ml_workspace(resource_group_name, aml_workspace_name)

    managed_id_principal_id, managed_id_client_id, managed_id_resource_id =\
        create_managed_identity(resource_group_name, managed_id_name)

    wait(seconds=60)

    grant_storage_account_permissions(managed_id_principal_id, storage_account_id)

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=aml_workspace_name,
    )

    create_compute_cluster(ml_client, compute_cluster_name, managed_id_client_id, managed_id_resource_id)
    create_environment(ml_client, environment_name)

    save_configuration_in_environment(subscription_id, resource_group_name, aml_workspace_name, storage_account_name,
                                      compute_cluster_name, managed_id_client_id, managed_id_resource_id,
                                      environment_name)


def confirm_used_subscription() -> str:
    action_text = 'Fetch details of currently selected subscription'
    start_action(action_text)

    subscription = execute_cli_command('az account show --query "{id:id,name:name}"')

    end_action(action_text)

    subscription_name = subscription['name']
    print(f'\nSubscription "{subscription_name}" will be used to create multiple resources.')
    response = input('Do you want to continue? [Y/n] ')

    if len(response) > 0 and response.lower() != 'y':
        print('\nExecute "az account set --name <name>" to select a different subscription')
        print('For more information visit ' +
              'https://learn.microsoft.com/en-us/cli/azure/account?view=azure-cli-latest#az-account-set')
        quit()

    return subscription['id']


def create_resource_group(resource_group_name: str):
    action_text = f'Create Resource Group "{resource_group_name}"'
    start_action(action_text)

    _ = execute_cli_command(f'az group create' +
                            f' --name {resource_group_name}' +
                            f' --location westeurope')
    end_action(action_text)


def create_azure_ml_workspace(resource_group_name: str, aml_workspace_name: str) -> (str, str):
    action_text = f'Create Azure Machine Learning Workspace "{aml_workspace_name}" and supportive resources'
    start_action(action_text)

    aml_workspace = execute_cli_command(f'az ml workspace create ' +
                                        f' --resource-group {resource_group_name}' +
                                        f' --name {aml_workspace_name}')

    storage_account_id = aml_workspace['storageAccount']
    storage_account_name = execute_cli_command(f'az storage account show' +
                                               f' --id {storage_account_id}' +
                                               f' --query "name"')

    end_action(action_text)
    return storage_account_id, storage_account_name


def create_managed_identity(resource_group: str, managed_id_name: str) -> (str, str, str):
    action_text = 'Create User-assigned Managed Identity'
    start_action(action_text)

    managed_id: dict = execute_cli_command(f'az identity create' +
                                           f' --resource-group {resource_group}' +
                                           f' --name {managed_id_name}')
    end_action(action_text)
    return managed_id['principalId'], managed_id['clientId'], managed_id['id']


def wait(seconds: int):
    action_text = f'Wait for {seconds} seconds'
    start_action(action_text)
    time.sleep(seconds)
    end_action(action_text)


def grant_storage_account_permissions(managed_id_principal_id: str, storage_account_id: str):
    role = 'Storage Blob Data Contributor'
    action_text = f'Grant User-assigned Managed Identity {role} on storage account'
    start_action(action_text)

    _ = execute_cli_command(f'az role assignment create' +
                            f' --assignee "{managed_id_principal_id}"' +
                            f' --role "{role}"' +
                            f' --scope "{storage_account_id}"')

    end_action(action_text)


def create_compute_cluster(ml_client: MLClient, compute_instance_name: str,
                           managed_id_client_id: str, managed_id_resource_id: str):
    action_text = 'Create compute cluster'
    start_action(action_text)
    if compute_instance_name not in [ci.name for ci in ml_client.compute.list()]:

        cpu_cluster = AmlCompute(
            name=compute_instance_name, type='amlcompute', size='Standard_DS3_v2', tier='Dedicated',
            min_instances=0, max_instances=2, idle_time_before_scale_down=300,
            identity=IdentityConfiguration(type='user_assigned', user_assigned_identities=[
                ManagedIdentityConfiguration(client_id=managed_id_client_id, resource_id=managed_id_resource_id)
            ])
        )

        ml_client.compute.begin_create_or_update(cpu_cluster)

        end_action(action_text)
    else:
        end_action('Skipped compute cluster creation', state='skipped')


def create_environment(ml_client: MLClient, environment_name: str):
    action_text = 'Create execution environment'
    start_action(action_text)

    pipeline_job_env = Environment(
        name=environment_name, description='Custom environment for digit classification',
        tags={'scikit-learn': '1.1.2'},
        conda_file=os.path.join('./', '1_conda_env.yml'),
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest'
    )

    ml_client.environments.create_or_update(pipeline_job_env)

    end_action(action_text)


def save_configuration_in_environment(subscription_id, resource_group_name, aml_workspace_name, storage_account_name,
                                      compute_cluster_name, managed_id_client_id, managed_id_resource_id,
                                      environment_name):
    action_text = 'Save configuration in .env file'
    start_action(action_text)

    dotenv.set_key('../.env', 'SUBSCRIPTION_ID', subscription_id)
    dotenv.set_key('../.env', 'RESOURCE_GROUP', resource_group_name)
    dotenv.set_key('../.env', 'AML_WORKSPACE_NAME', aml_workspace_name)
    dotenv.set_key('../.env', 'AML_STORAGE_ACCOUNT', storage_account_name)
    dotenv.set_key('../.env', 'COMPUTE_INSTANCE_NAME', compute_cluster_name)
    dotenv.set_key('../.env', 'COMPUTE_IDENTITY_CLIENT_ID', managed_id_client_id)
    dotenv.set_key('../.env', 'COMPUTE_IDENTITY_RESOURCE_ID', managed_id_resource_id)
    dotenv.set_key('../.env', 'ENVIRONMENT_NAME', environment_name)

    end_action(action_text)


if __name__ == '__main__':
    main()
