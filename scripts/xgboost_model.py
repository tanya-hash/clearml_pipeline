import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn.utils import class_weight

from clearml import Task, OutputModel
task = Task.init(project_name='UpSell_CrossSell_project', task_name='Model 1: XGBoost', task_type="training")


df = pd.read_csv("dataset/Processed_Data.csv")
df.drop(["Unnamed: 0"], axis=1, inplace=True)
print(df.head())

X = df.drop(["Response"], axis=1)
y = df.Response

print(X.shape, y.shape)

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

classes_weights=class_weight.compute_sample_weight(class_weight='balanced',y=y_train)
xgb = XGBClassifier(learning_rate=0.05)
params = xgb.get_params()
parameters = task.connect(params)
xgb.fit(X_train, y_train, sample_weight =classes_weights)
joblib.dump(xgb,"xgb_model.pkl")
predict = xgb.predict(X_test)

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

# plot_importance(xgb,max_num_features=10,height=0.4)

task.get_logger().report_confusion_matrix("Confusion Matrix","UpSell",matrix=cm,
    xaxis="Predicted Label",
    yaxis="True Label",
)

xg_probs = xgb.predict_proba(X_test)[:, 1]
roc_value = roc_auc_score(y_test, xg_probs)
fpr, tpr, thresholds = roc_curve(y_test, xg_probs)

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
