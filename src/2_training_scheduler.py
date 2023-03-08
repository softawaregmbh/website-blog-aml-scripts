import os

from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from utils import start_action, end_action

load_dotenv('.env')


def main():
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv('SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('RESOURCE_GROUP'),
        workspace_name=os.getenv('AML_WORKSPACE_NAME'),
    )

    print()
    action_text = 'Queue model training job'
    start_action(action_text)

    environment = f'{os.getenv("ENVIRONMENT_NAME")}@latest'
    compute_instance = os.getenv('COMPUTE_INSTANCE_NAME')
    job = command(code='./', command='python ./src/2_training.py', environment=environment, compute=compute_instance,
                  experiment_name='train_digit_classifier_model', display_name='Digit Classifier Model Training')

    ml_client.jobs.create_or_update(job)
    end_action(action_text)


if __name__ == '__main__':
    main()
