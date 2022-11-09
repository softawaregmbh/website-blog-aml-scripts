import os
import uuid

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
import dotenv

dotenv.load_dotenv('../.env')


def main():
    endpoint_name = f'digit-endpoint-{str(uuid.uuid4())[:8]}'
    deployment_name = 'blue'

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv('SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('RESOURCE_GROUP'),
        workspace_name=os.getenv('AML_WORKSPACE_NAME'),
    )

    scoring_uri = create_endpoint(ml_client, endpoint_name)

    deploy_model_to_endpoint(ml_client, endpoint_name, deployment_name)

    # TODO: Retrieve primary key API token for authorization against endpoint

    update_environment_configuration(scoring_uri, endpoint_name, deployment_name)


def create_endpoint(ml_client: MLClient, endpoint_name: str) -> str:
    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")

    print(f'\nCreating online endpoint {endpoint_name} and waiting for completion ...')

    long_running_operation = ml_client.online_endpoints.begin_create_or_update(endpoint)
    endpoint = long_running_operation.result()

    print('\t... endpoint created')
    return endpoint.scoring_uri


def deploy_model_to_endpoint(ml_client: MLClient, endpoint_name: str, deployment_name: str):
    model_name = os.getenv('MODEL_NAME')

    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=model_name)]
    )
    model = ml_client.models.get(name=model_name, version=str(latest_model_version))

    print('\nCreating new model deployment to online endpoint and waiting for completion ...')

    deployment = ManagedOnlineDeployment(name=deployment_name, endpoint_name=endpoint_name, model=model,
                                         instance_type="Standard_DS3_v2", instance_count=1)

    long_running_operation = ml_client.online_deployments.begin_create_or_update(deployment)
    long_running_operation.wait()

    print('\t... deployment created')


def update_environment_configuration(scoring_uri: str, endpoint_name: str, deployment_name: str):
    print('\nUpdating environment configuration')
    dotenv.set_key('../.env', 'ENDPOINT_URL', scoring_uri)
    dotenv.set_key('../.env', 'ENDPOINT_API_KEY', '<retrieve API key using provided comands>')
    dotenv.set_key('../.env', 'ENDPOINT_MODEL_DEPLOYMENT', deployment_name)

    print('\nExecute the following commands to retrieve the API key, then store it in the .env-backup file:')
    print('\n# Installs azure ml extension')
    print('az extension add -n ml\t\t')
    print('# Retrieve API key')
    print(f'az ml online-endpoint get-credentials -g {os.getenv("RESOURCE_GROUP")} -w {os.getenv("AML_WORKSPACE_NAME")}'
          f' -n {endpoint_name} -o tsv --query primaryKey')


if __name__ == '__main__':
    main()
