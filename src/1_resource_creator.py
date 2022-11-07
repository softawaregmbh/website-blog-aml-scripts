import os
import json
import subprocess
from typing import Union

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, IdentityConfiguration, ManagedIdentityConfiguration
from azure.identity import DefaultAzureCredential
import dotenv


def execute_cli_command(command: str) -> Union[str, list, dict]:
    command = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, shell=True)
    output, error_output = command.communicate()

    if command.returncode != 0:
        raise RuntimeError(error_output)

    if output.startswith('[') or output.startswith('{'):
        color_reset_code = '\x1b[0m'
        output = output[: -len(color_reset_code)] if output.endswith(color_reset_code) else output
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

    resource_group_name, subscription_id, aml_workspace_name = request_aml_workspace_info()

    managed_id_principal_id, managed_id_client_id, managed_id_resource_id =\
        create_managed_identity(resource_group_name, managed_id_name)

    storage_account_id, storage_account_name = fetch_storage_account_data(resource_group_name)

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


def request_aml_workspace_info() -> (str, str, str):
    print('\n\nEnter the resource group\'s name where the Azure Machine Learning workspace is located:')
    exists = False
    while exists is False:
        # TODO: improve with multi-select
        resource_group_name = input('Resource group> ')

        resource_groups: list[dict] = execute_cli_command('az group list' +
                                                          ' --query "[].{name:name,id:id}"')
        exists = resource_group_name in [rg['name'] for rg in resource_groups]
        if exists is False:
            print('\n\tThis resource group does not exist, please try again')

    print('')

    action_text = 'Gather AML workspace info'
    start_action(action_text)

    resource_group_id: str = [rg['id'] for rg in resource_groups if rg['name'] == resource_group_name][0]
    subscription_id = resource_group_id.split('/')[2]

    output = execute_cli_command(f'az ml workspace list' +
                                 f' --resource-group {resource_group_name}' +
                                 f' --query "[].name"')
    aml_workspace_name = output[0]

    end_action(action_text)

    return resource_group_name, subscription_id, aml_workspace_name


def create_managed_identity(resource_group: str, managed_id_name: str) -> (str, str, str):
    action_text = 'Create user-assigned managed identity'
    start_action(action_text)

    managed_id: dict = execute_cli_command(f'az identity create' +
                                           f' --resource-group {resource_group}' +
                                           f' --name {managed_id_name}')

    end_action(action_text)

    return managed_id['principalId'], managed_id['clientId'], managed_id['id']


def fetch_storage_account_data(resource_group_name: str) -> (str, str):
    action_text = 'Fetch storage accounts in resource group'
    start_action(action_text)

    storage_account_ids: list[dict] = execute_cli_command('az storage account list' +
                                                          f' --resource-group {resource_group_name}' +
                                                          '  --query "[].{id:id,name:name}"')

    end_action(action_text)

    if len(storage_account_ids) == 1:
        return storage_account_ids[0]['id'], storage_account_ids[0]['name']
    else:
        # TODO: pick single storage account id
        raise ValueError('not yet implemented!')


def grant_storage_account_permissions(managed_id_principal_id: str, storage_account_id: str):
    role = 'Storage Blob Data Contributor'
    action_text = f'Grant user-assigned managed identity {role} on storage account'
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
