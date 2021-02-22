Prerequisit: install mlflow in an environment, either conda or venv/virtualenv
             have python and pip installed  

To start a MLflow project:
prepare MLflow YAML file (as in 01-code_scripts/venv.YAML)</n>
prepare MLflow project file : as in 01-code_scripts/MLProject</n>
saved it in the file folder where codes run.

To check the logged MLflow artifacts(files, logs, images, models...):</n>
run in command line: </n>
$mlflow ui

default ui will be at host 0.0.0.0, and port 5000 </n>
So bring up a browser window and insert http://localhost:5000 </n>
While no experiments are created yet. You will see only 'default' as one experiment </n>
use UI click '+' to create new experiments : 'data_prep' and 'model_build' </n>
Specify where the path to a folder where you want to log your artifacts. </n>
Need to first create them, then run code with mlflow.start_run() or mlflow.log_xxxx in the code. </n>
So , in this specific case, my artifact folder is under 02-data/artifact_root/ </n>
and 04-model/artifact_root.




