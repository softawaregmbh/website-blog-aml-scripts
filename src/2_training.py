import os

from azureml.core import Workspace, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

load_dotenv('./.env')


def main():
    subscription_id = os.getenv('SUBSCRIPTION_ID')
    resource_group = os.getenv('RESOURCE_GROUP')
    workspace_name = os.getenv('AML_WORKSPACE_NAME')
    tenant_id = os.getenv('AMLW_CLIENT_TENANT_ID')
    app_id = os.getenv('AMLW_CLIENT_APP_ID')
    app_password = os.getenv('AMLW_CLIENT_PASSWORD')

    workspace = Workspace(subscription_id, resource_group, workspace_name, auth=ServicePrincipalAuthentication(
        tenant_id=tenant_id, service_principal_id=app_id, service_principal_password=app_password
    ))

    # enable debugging
    mlflow.start_run()
    mlflow.sklearn.autolog()

    # fetch train and test dataset
    train = Dataset.get_by_name(workspace, name='MNIST Database - Train Partition').to_pandas_dataframe()
    test = Dataset.get_by_name(workspace, name='MNIST Database - Test Partition').to_pandas_dataframe()

    x_train, x_test = train.loc[:, train.columns != 'label'], test.loc[:, test.columns != 'label']
    y_train, y_test = train[['label']].values.ravel(), test[['label']].values.ravel()

    # standard scaling
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    # define hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(100,), (125,), (100, 100)],
        'activation': ['logistic', 'relu'],
        'solver': ['lbfgs'],
        'alpha': [1E-4, 1E-3],

        'max_iter': [500],
    }

    # GridSearchCV
    param_tuner = GridSearchCV(MLPClassifier(), param_grid=param_grid, n_jobs=-1, cv=5, verbose=1)
    param_tuner.fit(x_train, y_train)
    digit_classifier = param_tuner.best_estimator_

    # predict on test set using best model from CV
    y_pred = digit_classifier.predict(x_test)
    print(classification_report(y_test, y_pred))

    model_name = os.getenv('MODEL_NAME')
    mlflow.sklearn.log_model(
        sk_model=digit_classifier,
        registered_model_name=model_name,
        artifact_path=model_name,
    )

    mlflow.sklearn.save_model(
        sk_model=digit_classifier,
        path=os.path.join(model_name, "trained_model"),
    )

    mlflow.end_run()


if __name__ == '__main__':
    main()
