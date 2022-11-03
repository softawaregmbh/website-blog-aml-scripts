import os

import dotenv

dotenv.load_dotenv('../.env')


def main():
    managed_identity_name = 'id-compute-cluster-test'

    print('\nThis script guides you through the setup self-assigned managed-identity process')
    print('\n\n1. Execute to create a self-assigned managed identity:')
    print(f'\naz identity create --resource-group {os.getenv("RESOURCE_GROUP")} --name {managed_identity_name}')
    input('\n\nEnter any character to continue...')

    print('\n\n2. Execute to retrieve the managed identity\'s data:')
    print(f'\naz identity list --query "[?name==\'{managed_identity_name}\'].{{clientId:clientId,prinipalId:principalId,resourceId:id}}"')
    managed_identity_client_id = input('\n\nEnter its client id: ')
    managed_identity_principal_id = input('\nEnter its principal id: ')
    managed_identity_resource_id = input('\nEnther its resource id:')

    dotenv.set_key('../.env', 'COMPUTE_IDENTITY_CLIENT_ID', managed_identity_client_id)
    dotenv.set_key('../.env', 'COMPUTE_IDENTITY_RESOURCE_ID', managed_identity_resource_id)

    print('\n\n3. Execute to retrieve the storage account\'s resource id:')
    print(f'\naz storage account list --query "[?name==\'{os.getenv("AML_STORAGE_ACCOUNT")}\'].id" -o tsv')
    storage_account_scope_id = input('\n\nEnter its resource id: ')

    print('\n\n4. Assign the "Storage Blob Data Contributor" role to the self-assigned managed identity"')
    print(f'\naz role assignment create --assignee "{managed_identity_principal_id}"'
          + f' --role "Storage Blob Data Contributor"'
          + f' --scope "{storage_account_scope_id}"')


if __name__ == '__main__':
    main()
