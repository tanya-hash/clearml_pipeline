from clearml import PipelineController
# import pip_system_certs
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.component(return_values=["xgb"],cache=True, task_type=TaskTypes.training, repo="https://github.com/tanya-hash/clearml_pipeline.git", repo_branch="dev")
def xgboost_train():
    print("<<<<<<<<<<<<<<<<<<<<Importing Modules>>>>>>>>>>>>>>>>>")
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    from sklearn.utils import class_weight

    df = pd.read_csv("dataset/Processed_Data.csv")
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    print(df.head())

    X = df.drop(["Response"], axis=1)
    y = df.Response

    print(X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    xgb = XGBClassifier(learning_rate=0.05)
    # params = xgb.get_params()
    xgb.fit(X_train, y_train, sample_weight=classes_weights)
    joblib.dump(xgb, "xgb_model.pkl")
    print("Completed XGB Training")
    return xgb

@PipelineDecorator.component(return_values=['rf_model','X_test','y_test'], cache=True, task_type=TaskTypes.training, repo="https://github.com/tanya-hash/clearml_pipeline.git", repo_branch="dev")
def rf_train():
    print("<<<<<<<<<<<Importing Modules>>>>>>>>>>>>>")
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib

    df = pd.read_csv("dataset/Processed_Data.csv")
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    print(df.head())

    X = df.drop(["Response"], axis=1)
    y = df.Response

    print(X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    rf_model = RandomForestClassifier(class_weight='balanced', max_depth=10,
                                      min_samples_leaf=20, min_samples_split=70,
                                      n_estimators=500, random_state=0)

    print("<<<<<<<<<<<< Training >>>>>>>>>>>>>>>>>>>")
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "RF_model.pkl")
    print("Completed Random Forest")
    return rf_model, X_test, y_test

@PipelineDecorator.component(return_values=["accuracy_xgb","accuracy_rf"], cache=True, task_type=TaskTypes.qc)
def inference(rf_model, xgb_model, X_test, y_test):
    from sklearn import metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, f1_score
    from clearml import task, Logger
    import matplotlib.pyplot as plt

    predict_rf = rf_model.predict(X_test)
    print("predict_rf", predict_rf)
    accuracy_rf = accuracy_score(y_test, predict_rf)
    Logger.current_logger().report_single_value(name="Random Forest Accuracy", value=accuracy_rf)
    precision_rf = precision_score(y_test, predict_rf)
    Logger.current_logger().report_single_value(name="Random Forest Precision", value=precision_rf)
    recall_rf = recall_score(y_test, predict_rf)
    Logger.current_logger().report_single_value(name="Random Forest Recall", value=recall_rf)
    f1_rf = f1_score(y_test, predict_rf)
    Logger.current_logger().report_single_value(name="Random Forest F1", value=f1_rf)

    cm = metrics.confusion_matrix(y_test, predict_rf)
    Logger.current_logger().report_confusion_matrix(
    "Random Forest",
    "ignored",
    matrix=cm,
    xaxis="Predicted",
    yaxis="True",
)

    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    roc_value = roc_auc_score(y_test, rf_probs)
    fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
    plt.figure()
    plot = plt.plot(fpr, tpr, label=' (area = %0.2f)' % roc_value)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' RF  ROC Curve')
    plt.legend(loc="lower right")
    plt.show(block=False)
    Logger.current_logger().report_matplotlib_figure(
    title="Random Forest ROC",
    series="ignored",
    figure=plot,
    report_interactive=True,
)

    predict_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, predict_xgb)
    Logger.current_logger().report_single_value(name="XGBoost Accuracy", value=accuracy_xgb)
    precision_xgb = precision_score(y_test, predict_xgb)
    Logger.current_logger().report_single_value(name="XGBoost Precision", value=precision_xgb)
    recall_xgb = recall_score(y_test, predict_xgb)
    Logger.current_logger().report_single_value(name="XGBoost Recall", value=recall_xgb)
    f1_xgb = f1_score(y_test, predict_xgb)
    Logger.current_logger().report_single_value(name="XGBoost F1", value=f1_xgb)

    cm = metrics.confusion_matrix(y_test, predict_xgb)
    Logger.current_logger().report_confusion_matrix(
    "XGBoost",
    "ignored",
    matrix=cm,
    xaxis="Predicted",
    yaxis="True",
)

    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    roc_value = roc_auc_score(y_test, xgb_probs)
    fpr, tpr, thresholds = roc_curve(y_test, xgb_probs)
    plt.figure()
    plot = plt.plot(fpr, tpr, label=' (area = %0.2f)' % roc_value)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' RF  ROC Curve')
    plt.legend(loc="lower right")
    plt.show(block=False)
    Logger.current_logger().report_matplotlib_figure(
    title="XGBoost ROC",
    series="ignored",
    figure=plot,
    report_interactive=True,
)

    print("accuracies",accuracy_xgb, accuracy_rf)

    return accuracy_xgb,accuracy_rf


@PipelineDecorator.pipeline(name="Upsell_CrossSell_pipeline", project="examples", version="0.0.5", pipeline_execution_queue=None)
def executing_pipeline(mock_parameter="mock", xgb_model=None, rf_model=None):
    print(mock_parameter)

    # Use the pipeline argument to start the pipeline and pass it ot the first step
    print("<<<<<<<<<<launch step one>>>>>>>>>>")
    xgb_model = xgboost_train()

    print("<<<<<<<<<<launch step two>>>>>>>>>>")
    rf_model, X_test, y_test = rf_train()

    print("<<<<<launch step three>>>>>>")
    accuracy_xgb, accuracy_rf = inference(rf_model,xgb_model, X_test, y_test)

if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('clearml-demo')
    PipelineDecorator.debug_pipeline()
    executing_pipeline()

    print("process completed")
