import mlflow
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Any

#mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

def create_experiment(  experiment_name: str, artifact_location: str, tags: dict[str, Any]):
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    return experiment_id


experiment_id=create_experiment(experiment_name="testing_mlflow1",artifact_location="testing_mlflow1_artifacts", tags={"env": "mlflowenv", "version": "1.0.0"})
mlflow.set_experiment(experiment_id=experiment_id)

if experiment_id is not None:
    experiment = mlflow.get_experiment(experiment_id)
elif experiment_name is not None:
    experiment = mlflow.get_experiment_by_name(experiment_name)

#experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

#mlflow.set_experiment(experiment_name=experiment_name)

print("Name: {}".format(experiment.name))
print("ID: {}".format(experiment.experiment_id))
with mlflow.start_run(run_name="logging_images", experiment_id=experiment.experiment_id) as run:

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    # log the precision-recall curve
    fig_pr = plt.figure()
    pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Precision-Recall Curve")
    plt.legend()

    mlflow.log_figure(fig_pr, "metrics/precision_recall_curve.png")

    # log the ROC curve
    fig_roc = plt.figure()
    roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("ROC Curve")
    plt.legend()

    mlflow.log_figure(fig_roc, "metrics/roc_curve.png")

    # log the confusion matrix
    fig_cm = plt.figure()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.legend()

    mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")

    # print info about the run
    print("run_id: {}".format(run.info.run_id))
    print("experiment_id: {}".format(run.info.experiment_id))
    print("status: {}".format(run.info.status))
    print("start_time: {}".format(run.info.start_time))
    print("end_time: {}".format(run.info.end_time))
    print("lifecycle_stage: {}".format(run.info.lifecycle_stage))