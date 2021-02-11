from azureml.core import Workspace

subscription_id = '9b46252f-8ad8-4a4e-a607-74a46d555a2e'
resource_group  = 'ganchan'
workspace_name  = 'ganchan'

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ws.write_config()
    print('Library configuration succeeded')
except:
    print('Workspace not found')