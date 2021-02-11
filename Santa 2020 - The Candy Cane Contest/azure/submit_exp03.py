import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

import os

experiment_folder = 'hyperdrive03'
os.makedirs(experiment_folder, exist_ok=True)

print('Folder ready.')

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# cluster_name = "HighCPUcluster"
cluster_name = "LP-HCPUcluster"


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
sklearn_env = Environment("env02")

# Ensure the required packages are installed (we need scikit-learn, Azure ML defaults, and Azure ML dataprep)
packages = CondaDependencies.create(pip_packages=['lightgbm','sklearn','scipy','numpy','azureml-defaults','azureml-dataprep[pandas]'])
sklearn_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                              script='training03.py',
                              arguments = [
                                    '--max_depth', 5,
                                    '--num_leaves',50, 
                                    '--subsample',0.9,
                                    '--learning_rate',0.01,
                                    '--min_data_in_leaf', 50,
                                    '--lambda_l1',20,
                                    '--lambda_l2',20,
                                    '--n_estimators',1000
                                    ], 
                              environment=sklearn_env,
                              compute_target = training_cluster)

# Sample a range of parameter values
params = BayesianParameterSampling(
    {
        # There's only one parameter, so grid sampling will try each value - with multiple parameters it would try every combination
        '--max_depth': choice(list(range(2,20))),
        '--num_leaves':choice(list(range(6,251))), 
        '--subsample':uniform(0.5,1),
        '--learning_rate':uniform(0.005,0.25),
        '--min_data_in_leaf': choice(list(range(2,501))),
        '--lambda_l1': choice(list(range(201))),
        '--lambda_l2': choice(list(range(201))),
        '--n_estimators': choice(list(range(100,4001,100)))
    }
)

# Configure hyperdrive settings
hyperdrive = HyperDriveConfig(run_config=script_config, 
                          hyperparameter_sampling=params, 
                          policy=None, 
                          primary_metric_name='rmse', 
                          primary_metric_goal=PrimaryMetricGoal.MINIMIZE, 
                          max_total_runs=160,
                          max_concurrent_runs=4)

# Run the experiment
experiment = Experiment(workspace = ws, name = 'training_hyperdrive03')
run = experiment.submit(config=hyperdrive)
print("Experiment is running...")

# Show the status in the notebook as the experiment runs
# RunDetails(run).show()
run.wait_for_completion()
print("Experiment has done.")