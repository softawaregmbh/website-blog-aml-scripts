import os
import uuid

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
import dotenv

from utils import execute_cli_command, start_action, end_action

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

    print()
    scoring_uri = create_endpoint(ml_client, endpoint_name)

    deploy_model_to_endpoint(ml_client, endpoint_name, deployment_name)

    api_key = fetch_endpoint_api_key(endpoint_name)

    update_environment_configuration(scoring_uri, deployment_name, api_key)


def create_endpoint(ml_client: MLClient, endpoint_name: str) -> str:
    action_text = f'Create online endpoint "{endpoint_name}"'
    start_action(action_text)

    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")

    long_running_operation = ml_client.online_endpoints.begin_create_or_update(endpoint)
    endpoint = long_running_operation.result()

    end_action(action_text)
    return endpoint.scoring_uri


def deploy_model_to_endpoint(ml_client: MLClient, endpoint_name: str, deployment_name: str):
    model_name = os.getenv('MODEL_NAME')
    action_text = f'Deploy model "{model_name}" to endpoint "{endpoint_name}"'
    start_action(action_text)

    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=model_name)]
    )
    model = ml_client.models.get(name=model_name, version=str(latest_model_version))

    deployment = ManagedOnlineDeployment(name=deployment_name, endpoint_name=endpoint_name, model=model,
                                         instance_type="Standard_DS3_v2", instance_count=1)

    long_running_operation = ml_client.online_deployments.begin_create_or_update(deployment)
    long_running_operation.wait()

    end_action(action_text)


def fetch_endpoint_api_key(endpoint_name: str) -> str:
    action_text = f'Fetch API key of Endpoint "{endpoint_name}"'
    start_action(action_text)

    api_key = execute_cli_command(f'az ml online-endpoint get-credentials' +
                                  f' --resource-group {os.getenv("RESOURCE_GROUP")}' +
                                  f' --workspace-name {os.getenv("AML_WORKSPACE_NAME")}' +
                                  f' --name {endpoint_name}' +
                                  f' --query "primaryKey"')
    end_action(action_text)
    return api_key


def update_environment_configuration(scoring_uri: str, deployment_name: str, api_key: str):
    action_text = 'Update environment configuration'
    start_action(action_text)

    dotenv.set_key('../.env', 'ENDPOINT_URL', scoring_uri)
    dotenv.set_key('../.env', 'ENDPOINT_API_KEY', api_key)
    dotenv.set_key('../.env', 'ENDPOINT_MODEL_DEPLOYMENT', deployment_name)

    end_action(action_text)


if __name__ == '__main__':
    main()
