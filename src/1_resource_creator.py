import json
import os
import subprocess
from typing import Union

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, IdentityConfiguration, ManagedIdentityConfiguration
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv('../.env')


def execute_cli_command(command: str) -> Union[str, list, dict]:
    command = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, shell=True)
    output, error = command.communicate()

    if error:
        raise RuntimeError(error)

    if output.startswith('[') or output.startswith('{'):
        return json.loads(output)
    else:
        return output


def start_action(action_text: str):
    print(f'⚪ {action_text}', end='')


def end_action(action_text: str, state: str = 'success'):
    if state == 'success':
        status_symbol = '🟢'
    elif state == 'skipped':
        status_symbol = '🔵'
    else:
        raise ValueError(f'State {state} unhandled.')

    print(f'\r{status_symbol} {action_text}')


def main():
    managed_id_name = 'id-compute-cluster-test'
    compute_cluster_name = 'cluster-standard-ds3-v2'
    environment_name = 'env-digit-classifier'

    resource_group_name = request_resource_group_name()
    print('\n')

    managed_id_principal_id, managed_id_client_id, managed_id_resource_id =\
        create_managed_identity(resource_group_name, managed_id_name)

    storage_account_id = fetch_storage_account_id(resource_group_name)

    grant_storage_account_permissions(managed_id_principal_id, storage_account_id)

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv('SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('RESOURCE_GROUP'),
        workspace_name=os.getenv('AML_WORKSPACE_NAME'),
    )

    create_compute_cluster(ml_client, compute_cluster_name, managed_id_client_id, managed_id_resource_id)
    create_environment(ml_client, environment_name)

    # todo: Update .env file


def request_resource_group_name() -> str:
    print('\n\nEnter the resource group\'s name where the Azure Machine Learning workspace is located:')
    exists = False
    while exists is False:
        # TODO: improve with multi-select
        resource_group = input('\nResource group> ')
        exists = resource_group in execute_cli_command('az group list --query "[].name"')
        if exists is False:
            print('\n\tThis resource group does not exist, please try again')

    return resource_group


def create_managed_identity(resource_group: str, managed_id_name: str) -> (str, str, str):
    action_text = 'Create user-assigned managed identity'
    start_action(action_text)

    managed_id: dict = execute_cli_command(f'az identity create ' +
                                           f'--resource-group {resource_group} ' +
                                           f'--name {managed_id_name}')

    end_action(action_text)

    return managed_id['principalId'], managed_id['clientId'], managed_id['id']


def fetch_storage_account_id(resource_group_name: str) -> str:
    action_text = 'Fetch storage account id list'
    start_action(action_text)

    storage_account_ids = execute_cli_command(f'az storage account list ' +
                                              f'--resource-group {resource_group_name} ' +
                                              f'--query "[].id"')

    end_action(action_text)

    if len(storage_account_ids) == 1:
        return storage_account_ids[0]
    else:
        # TODO: pick single storage account id
        raise ValueError('not yet implemented!')


def grant_storage_account_permissions(managed_id_principal_id: str, storage_account_id: str):
    role = 'Storage Blob Data Contributor'
    action_text = f'Grant user-assigned managed identity {role} on storage account'
    start_action(action_text)

    _ = execute_cli_command(f'az role assignment create ' +
                            f'--assignee "{managed_id_principal_id}" ' +
                            f'--role "{role}" ' +
                            f'--scope "{storage_account_id}"')

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


if __name__ == '__main__':
    main()
