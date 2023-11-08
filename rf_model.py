import pandas as pd
# from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
import joblib

from clearml import Task
task = Task.init(project_name='UpSell_CrossSell_project', task_name='Model 3: Random Forest', task_type="training")


df = pd.read_csv("dataset/Processed_Data.csv")
df.drop(["Unnamed: 0"], axis=1, inplace=True)
print(df.head())

X = df.drop(["Response"], axis=1)
y = df.Response

print(X.shape, y.shape)

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

rf_model=RandomForestClassifier(class_weight='balanced', max_depth=10,
                       min_samples_leaf=20, min_samples_split=70,
                       n_estimators=500, random_state=0)
params = rf_model.get_params()
parameters = task.connect(params)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model,"RF_model.pkl")
predict = rf_model.predict(X_test)

acurracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)

task.get_logger().report_single_value(name="Accuracy", value=acurracy)
task.get_logger().report_single_value(name="Precision", value=precision)
task.get_logger().report_single_value(name="Recall", value=recall)
task.get_logger().report_single_value(name="F1-score", value=f1)

cm = metrics.confusion_matrix(y_test, predict)
print(cm)

task.get_logger().report_confusion_matrix("Confusion Matrix","UpSell",matrix=cm,
    xaxis="Predicted Label",
    yaxis="True Label",
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
plt.show()
task.get_logger().report_matplotlib_figure("ROC_AUC_Curve","UpSell_CrossSell_project",plot,report_image=False,report_interactive=True)
task.close()
