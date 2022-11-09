import os
import time
import uuid

import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, IdentityConfiguration, ManagedIdentityConfiguration
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
import dotenv
from sklearn import datasets

from utils import execute_cli_command, start_action, end_action, wait


def main():
    unique_suffix = str(uuid.uuid4())[:8]
    resource_group_name = f'rg-azure-ml-showcase-{unique_suffix}'
    aml_workspace_name = f'mlw-mlshowcase-{unique_suffix}'
    managed_id_name = 'id-compute-cluster'
    compute_cluster_name = 'cluster-standard-ds3-v2'
    environment_name = 'env-digit-classifier'
    app_registration_name = f'ar-aml-showcase-client-{unique_suffix}'

    # TODO: Check if azure cli is installed
    # TODO: Check if azure cli ml extension is enabled

    print()
    subscription_id = confirm_used_subscription()
    print()

    create_resource_group(resource_group_name)

    aml_workspace_id, storage_account_id, storage_account_name =\
        create_azure_ml_workspace(resource_group_name, aml_workspace_name)

    managed_id_principal_id, managed_id_client_id, managed_id_resource_id =\
        create_managed_identity(resource_group_name, managed_id_name)

    app_reg_id, app_reg_service_principal_id, app_reg_tenant_id, app_reg_app_id, app_reg_password =\
        create_app_registration(app_registration_name)

    wait(seconds=60)

    grant_permissions('User-assigned Managed Identity', managed_id_principal_id,
                      'AML Storage Account', storage_account_id, 'Storage Blob Data Contributor')
    grant_permissions('App Registration', app_reg_app_id,
                      'AML Workspace', aml_workspace_id, 'Contributor')

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=aml_workspace_name,
    )

    train_set, test_set = fetch_mnist_dataset()
    register_mnist_dataset(subscription_id, resource_group_name, aml_workspace_name, app_reg_tenant_id,
                           app_reg_app_id, app_reg_password, train_set, test_set)

    create_compute_cluster(ml_client, compute_cluster_name, managed_id_client_id, managed_id_resource_id)

    create_environment(ml_client, environment_name)

    save_configuration_in_environment(subscription_id, resource_group_name, aml_workspace_name, storage_account_name,
                                      compute_cluster_name, environment_name, app_reg_id, app_reg_tenant_id,
                                      app_reg_app_id, app_reg_password)


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

    aml_workspace_id = execute_cli_command(f'az ml workspace show' +
                                           f' --resource-group "{resource_group_name}"' +
                                           f' --name "{aml_workspace_name}"' +
                                           f' --query "id"')

    storage_account_id = aml_workspace['storageAccount']
    storage_account_name = execute_cli_command(f'az storage account show' +
                                               f' --id {storage_account_id}' +
                                               f' --query "name"')

    end_action(action_text)
    return aml_workspace_id, storage_account_id, storage_account_name


def create_managed_identity(resource_group: str, managed_id_name: str) -> (str, str, str):
    action_text = 'Create User-assigned Managed Identity'
    start_action(action_text)

    managed_id: dict = execute_cli_command(f'az identity create' +
                                           f' --resource-group {resource_group}' +
                                           f' --name {managed_id_name}')
    end_action(action_text)
    return managed_id['principalId'], managed_id['clientId'], managed_id['id']


def create_app_registration(app_registration_name: str) -> (str, str, str, str):
    action_text = f'Create App Registration "{app_registration_name}" to authorize model training later'
    start_action(action_text)

    app_registration_id = execute_cli_command(f'az ad app create' +
                                              f' --display-name {app_registration_name}' +
                                              f' --query "id"')

    app_registration_secrets = execute_cli_command(f'az ad app credential reset' +
                                                   f' --id {app_registration_id}' +
                                                   f' --append')

    app_registration_service_principal = execute_cli_command(f'az ad sp create' +
                                                             f' --id {app_registration_id}' +
                                                             f' --query "id"')

    end_action(action_text)

    return app_registration_id, app_registration_service_principal,\
           app_registration_secrets['tenant'], app_registration_secrets['appId'], app_registration_secrets['password']


def grant_permissions(assignee_title: str, assignee: str, scope_title: str, scope: str, role: str):
    action_text = f'Grant "{role}" role to "{assignee_title}" on "{scope_title}"'
    start_action(action_text)

    _ = execute_cli_command(f'az role assignment create' +
                            f' --assignee "{assignee}"' +
                            f' --role "{role}"' +
                            f' --scope "{scope}"')

    end_action(action_text)


def fetch_mnist_dataset() -> (pd.DataFrame, pd.DataFrame):
    action_text = 'Fetch MNIST dataset'
    start_action(action_text)

    data, labels = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    data['label'] = labels

    train_set, test_set = data.loc[:59999], data.loc[60000:]

    end_action(action_text)
    return train_set, test_set


def register_mnist_dataset(subscription_id, resource_group, workspace_name, tenant_id, app_id, app_password,
                           train_set, test_set):
    action_text = 'Register MNIST dataset in AML Workspace'
    start_action(action_text)

    workspace = Workspace(subscription_id, resource_group, workspace_name, auth=ServicePrincipalAuthentication(
        tenant_id=tenant_id, service_principal_id=app_id, service_principal_password=app_password
    ))
    datastore = Datastore.get(workspace, 'workspaceblobstore')

    _ = Dataset.Tabular.register_pandas_dataframe(train_set, datastore, 'MNIST Database - Train Partition',
                                                  show_progress=False)
    _ = Dataset.Tabular.register_pandas_dataframe(test_set, datastore, 'MNIST Database - Test Partition',
                                                  show_progress=False)

    end_action(action_text)


def create_compute_cluster(ml_client: MLClient, compute_instance_name: str,
                           managed_id_client_id: str, managed_id_resource_id: str):
    action_text = 'Create Compute Cluster in AML Workspace'
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
    action_text = 'Create Environment in AML Workspace'
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
                                      compute_cluster_name, environment_name, app_reg_id, app_reg_tenant_id,
                                      app_reg_app_id, app_reg_password):
    action_text = 'Save configuration in .env file'
    start_action(action_text)

    dotenv.set_key('../.env', 'SUBSCRIPTION_ID', subscription_id)
    dotenv.set_key('../.env', 'RESOURCE_GROUP', resource_group_name)
    dotenv.set_key('../.env', 'AML_WORKSPACE_NAME', aml_workspace_name)
    dotenv.set_key('../.env', 'AML_STORAGE_ACCOUNT', storage_account_name)
    dotenv.set_key('../.env', 'COMPUTE_INSTANCE_NAME', compute_cluster_name)
    dotenv.set_key('../.env', 'ENVIRONMENT_NAME', environment_name)
    dotenv.set_key('../.env', 'AMLW_CLIENT_ID', app_reg_id)
    dotenv.set_key('../.env', 'AMLW_CLIENT_TENANT_ID', app_reg_tenant_id)
    dotenv.set_key('../.env', 'AMLW_CLIENT_APP_ID', app_reg_app_id)
    dotenv.set_key('../.env', 'AMLW_CLIENT_PASSWORD', app_reg_password)
    dotenv.set_key('../.env', 'MODEL_NAME', 'digit-classifier')

    end_action(action_text)


if __name__ == '__main__':
    main()
