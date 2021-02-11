import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

import os

experiment_folder = 'make_nn'
os.makedirs(experiment_folder, exist_ok=True)

print('Folder ready.')

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# cluster_name = "HighCPUcluster"
cluster_name = "LP-HMemcluster"

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)

from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.train.hyperdrive import BayesianParameterSampling,uniform
from azureml.widgets import RunDetails

# Create a Python environment for the experiment
nn_env = Environment("env_nn")

# Ensure the required packages are installed (we need scikit-learn, Azure ML defaults, and Azure ML dataprep)
packages = CondaDependencies.create(pip_packages=['torch','sklearn','scipy','numpy','azureml-defaults','azureml-dataprep[pandas]'])
nn_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                              script='train_nn.py',
                              arguments = [
                                    '--num_epoch', 50,
                                    '--batch_size',100000,
                                    '--learning_rate',0.01,
                                    '--drop_rate',0.5
                                    ], 
                              environment=nn_env,
                              compute_target = training_cluster)

# Run the experiment
experiment = Experiment(workspace = ws, name = 'training_nn')
run = experiment.submit(config=script_config)
print("Experiment is running...")

# Show the status in the notebook as the experiment runs
# RunDetails(run).show()
run.wait_for_completion()
print("Experiment has done.")