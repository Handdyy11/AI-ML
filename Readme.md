# SENTRIX

## Run the tracked data and model pipeline

From PowerShell, activate the virtual environment and run each phase from the
`SENTRIX` directory:

```powershell
cd C:\Users\harsh\OneDrive\Desktop\cyber_defence_ai
.\.venv\Scripts\Activate.ps1
cd .\SENTRIX

python -m src.p2_preprocess.clean
python -m src.p2_preprocess.prepare
python -m src.p3_supervised.models
```

The cleaning and preparation runs record data counts, sampling/split settings,
class proportions, preprocessing statistics, the scaler, feature list, and
histograms. The model comparison run records aggregate model settings and
metrics, the k-NN tuning table, evaluation chart, and saved models. It also
creates a nested run for each fitted classifier (k-NN, decision tree, random
forest, and the majority-class baseline), with that model's parameters, test
metrics, and a loadable scikit-learn model artifact. These model artifacts
expect scaled input features; the scaler and selected feature list are logged
by the preparation run.

Runs are stored locally in a SQLite database and artifacts in
`SENTRIX\mlruns` under the `SENTRIX` experiment. This tracking directory is
local and ignored by Git. To view the runs, open a second PowerShell window:

```powershell
cd C:\Users\harsh\OneDrive\Desktop\cyber_defence_ai
.\.venv\Scripts\Activate.ps1
cd .\SENTRIX
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then browse to <http://127.0.0.1:5000>. Keep the terminal running the UI open
while you use it.