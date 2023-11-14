from clearml import PipelineController
# import pip_system_certs
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.component(return_values=["xgb"],cache=True, task_type=TaskTypes.training, repo="https://github.com/tanya-hash/clearml_pipeline.git", repo_branch="dev")
def xgboost_train():
    print("<<<<<<<<<<<<<<<<<<<<Importing Modules>>>>>>>>>>>>>>>>>")
    import pandas as pd
    from xgboost import XGBClassifier
    # from sklearn import metrics
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score, roc_auc_score
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
    # predict = xgb.predict(X_val)

    # accuracy = accuracy_score(y_val, predict)
    # precision = precision_score(y_val, predict)
    # recall = recall_score(y_val, predict)
    # f1 = f1_score(y_val, predict)
    #
    # cm = metrics.confusion_matrix(y_val, predict)
    # print(cm)
    #
    #
    # xg_probs = xgb.predict_proba(X_val)[:, 1]
    # roc_value = roc_auc_score(y_val, xg_probs)
    # fpr, tpr, thresholds = roc_curve(y_val, xg_probs)
    #
    print("Completed")
    return xgb

@PipelineDecorator.component(return_values=['rf_model','X_test','y_test'], cache=True, task_type=TaskTypes.training, repo="https://github.com/tanya-hash/clearml_pipeline.git", repo_branch="dev")
def rf_train():
    print("<<<<<<<<<<<Importing Modules>>>>>>>>>>>>>")
    import pandas as pd
    # from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import accuracy_score, roc_auc_score
    # from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score
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
    # params = rf_model.get_params()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "RF_model.pkl")
    # predict = rf_model.predict(X_val)
    #
    # accuracy = accuracy_score(y_val, predict)
    # precision = precision_score(y_val, predict)
    # recall = recall_score(y_val, predict)
    # f1 = f1_score(y_val, predict)
    #
    # cm = metrics.confusion_matrix(y_val, predict)
    # print(cm)
    #
    # rf_probs = rf_model.predict_proba(X_val)[:, 1]
    # roc_value = roc_auc_score(y_val, rf_probs)
    # fpr, tpr, thresholds = roc_curve(y_val, rf_probs)

    print("Completed")
    return rf_model, X_test, y_test

@PipelineDecorator.component(return_values=["accuracy_xgb","accuracy_rf"], cache=True, task_type=TaskTypes.qc)
def inference(rf_model, xgb_model, X_test, y_test):
    # from sklearn.linear_model import LogisticRegression  # noqa
    from sklearn.metrics import accuracy_score
    predict_rf = rf_model.predict(X_test)
    print("predict_rf", predict_rf)
    accuracy_rf = accuracy_score(y_test, predict_rf)

    predict_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, predict_xgb)

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
