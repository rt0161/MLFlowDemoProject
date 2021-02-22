import pandas as pd
import mlflow
logged_model = '../04-model/artifact_root/fa9b312af0824f29a1842af2cf3c550d/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# load previously logged data 
X_test=pd.read_csv('../02-data/data_model_built_xtest_set.csv', header =0)
y_test = pd.read_csv('../02-data/data_model_built_ytest_set.csv', header = 0)

y_pred = loaded_model.predict(X_test)
y_diff = y_test.price.values-y_pred

print(y_diff)
