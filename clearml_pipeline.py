from clearml import PipelineController
# import pip_system_certs
from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.component(return_values=['accuracy', 'precision', 'recall', 'f1'], cache=True)
def prediction():
    # make sure we have scikit-learn for this step, we need it to use to unpickle the object
    print(">>>>>>>>>>>Importing Modules>>>>>>>>>>>>>>>>>>>>>>>")
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score
    import pickle

    print(">>>>>>>>>>>>>Predicting..>>>>>>>>>>>>>>>>")
    with open("RF_model.pkl", 'rb') as f:
        model = pickle.load(f)
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")

    predict = model.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    print(">>>>>>>>>>Accuracy>>>>>>>>>>", accuracy)
    precision = precision_score(y_test, predict)
    recall = recall_score(y_test, predict)
    f1 = f1_score(y_test, predict)
    print(">>>>>>>>>>f1>>>>>>>>>>", f1)

    return accuracy, precision, recall, f1

if __name__ == '__main__':
        # create the pipeline controller
        print(">>>>>>>>>>Starting pipeline controller>>>>>>>>>")
        pipe = PipelineController(
            project='Upsell-Cross',
            name='Pipeline demo',
            version='1.1',
            add_pipeline_tags=False,
            repo = "https://github.com/tanya-hash/clearml_pipeline.git",
            repo_branch = "main",
            # repo_commit= "10cedf54e04d5acc4b279e03fddb8034b4828a94",

        )
        print(">>>>>>>pipeline controller ended>>>>>>>>>>>>>>>>>>")
        # set the default execution queue to be used (per step we can override the execution)
        pipe.set_default_execution_queue('clearml-demo')

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Starting Task 1")

        pipe.add_function_step(
            name='prediction',
            function= prediction,
            function_return=['accuracy', 'precision', 'recall', 'f1'],
            cache_executed_step=True,
        )

        # Start the pipeline on the services queue (remote machine, default on the clearml-server)
        pipe.start(queue="clearml-demo")

        print('pipeline completed')
