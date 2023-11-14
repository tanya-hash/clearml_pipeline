from clearml import PipelineController
# import pip_system_certs
from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.component(return_values=['dataframe'], cache=True)
def preprocessing():
    import pandas as pd
    from clearml import Task, Dataset

    dataset = Dataset.get(dataset_name="Raw_Data",
                          dataset_project="UpSell-Dataset")

    train = pd.read_csv(dataset)
    print("Preprocessing..")

    print(train.head())
    print(train["Response"].value_counts())
    print(train.shape)

    d = []
    for i in train['Age']:
        if i >= 20 and i <= 32:
            d.append('20-32')
        elif i >= 33 and i <= 52:
            d.append('33-52')
        elif i >= 53 and i <= 65:
            d.append('53-65')
        else:
            d.append('>65')

    u = pd.DataFrame(d, columns=['binn_age'])
    train = pd.concat([u, train], axis=1)
    train.drop(["Age"], axis=1, inplace=True)
    print(train.head())

    num_cols = ['Region_Code', 'Annual_Premium', 'Vintage']

    def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 3 * interquantile_range
        low_limit = quartile1 - 3 * interquantile_range
        return low_limit, up_limit

    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False

    for col in num_cols:
        print(f"{col} : {check_outlier(train, col)}")

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    for col in num_cols:
        replace_with_thresholds(train, col)

    train['Gender'] = train['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

    region = train[['Region_Code']].value_counts()
    print(region)
    reg_res = []
    for i in region.index:
        res = train[train['Region_Code'] == i]['Response'].value_counts()
        if 1 in res.index:
            reg_res.append(res[1])
        else:
            reg_res.append(0)

    percent = []
    for i in range(len(reg_res)):
        res = int(reg_res[i] / region.values[i] * 100)
        percent.append(res)

    region_code = pd.concat([region.reset_index(), pd.DataFrame(reg_res, columns=['reg_response']),
                             pd.DataFrame(percent, columns=['percentage'])], axis=1)
    region_code.columns = ['Region_Code', 'sum', 'reg_response', 'percentage']

    reg_code = []
    for i in train['Region_Code']:
        res = region_code[region_code['Region_Code'] == i]['percentage'].values[0]
        if res <= 5:
            reg_code.append('less than 5')  # A class
        elif res > 5 and res <= 10:
            reg_code.append('5-10')
        elif res > 10 and res <= 15:
            reg_code.append('10-15')
        else:
            reg_code.append('greater than 15')

    train = pd.concat([pd.DataFrame(reg_code, columns=['binn_region_code']), train], axis=1)
    train = train.drop('id', axis=1)

    print(train.head())

    psc = train['Policy_Sales_Channel'].value_counts()
    print(psc)
    psc_res = []
    for i in psc.index:
        res = train[train['Policy_Sales_Channel'] == i]['Response'].value_counts()
        if 1 in res.index:
            psc_res.append(res[1])
        else:
            psc_res.append(0)

    per = []
    for i in range(len(psc_res)):
        res = int(psc_res[i] / psc.values[i] * 100)
        per.append(res)

    policy_chh = pd.concat(
        [psc.reset_index(), pd.DataFrame(psc_res, columns=['psc_res']), pd.DataFrame(per, columns=['Percentage'])],
        axis=1)
    print(policy_chh.head())

    pol_sal_chh = []
    for i in train['Policy_Sales_Channel']:
        res = policy_chh[policy_chh['Policy_Sales_Channel'] == i]['Percentage'].values[0]
        if res <= 5:
            pol_sal_chh.append('less than 5')
        elif res > 5 and res <= 10:
            pol_sal_chh.append('5-10')
        elif res > 10 and res <= 15:
            pol_sal_chh.append('10-15')
        elif res > 15 and res <= 20:
            pol_sal_chh.append('15-20')
        elif res > 20 and res <= 25:
            pol_sal_chh.append('20-25')
        else:
            pol_sal_chh.append('greater than 25')

    train = pd.concat([pd.DataFrame(pol_sal_chh, columns=['binn_policy_sales_channel']), train], axis=1)
    print(train.head())

    v = []
    for i in train['Vintage']:
        if i <= 50:
            v.append('<=50')
        elif i >= 51 and i <= 100:
            v.append('51-100')
        elif i >= 101 and i <= 150:
            v.append('101-150')
        elif i >= 151 and i <= 200:
            v.append('151-200')
        elif i >= 201 and i <= 250:
            v.append('201-250')
        else:
            v.append('>250')

    vintage = pd.DataFrame(v, columns=['binn_vintage'])
    train = pd.concat([vintage, train], axis=1)
    train = train.drop(columns=['Region_Code', 'Policy_Sales_Channel', 'Vintage'])

    print(train.head())

    train['binn_vintage'] = train['binn_vintage'].map(
        {'51-100': "A", '101-150': "B", "151-200": "C", "201-250": "D", ">250": "E", "<=50": "F"})
    train['binn_policy_sales_channel'] = train['binn_policy_sales_channel'].map(
        {'15-20': "A", 'less than 5': "B", "20-25 ": "C", "greater than 25": "D", "10-15": "E", "5-10": "F"})
    train['binn_region_code'] = train['binn_region_code'].map(
        {'5-10': "A", 'greter than 15': "B", "10-15": "C", "less than 5": "D"})
    train['Vehicle_Age'] = train['Vehicle_Age'].map({'1-2 Year': "A", "< 1 Year": "B", "> 2 Years": "C"})

    new_df = pd.get_dummies(train, drop_first=True)

    return new_df

@PipelineDecorator.component(return_values=['accuracy','precision','recall','f1'], cache=True)
def xgboost_train():
    print("<<<<<<<<<<<<<<<<<<<<Importing Modules>>>>>>>>>>>>>>>>>")
    import pandas as pd
    # from matplotlib import pyplot as plt
    from xgboost import XGBClassifier
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score, roc_auc_score
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
    params = xgb.get_params()
    xgb.fit(X_train, y_train, sample_weight=classes_weights)
    joblib.dump(xgb, "xgb_model.pkl")
    predict = xgb.predict(X_val)

    accuracy = accuracy_score(y_val, predict)
    precision = precision_score(y_val, predict)
    recall = recall_score(y_val, predict)
    f1 = f1_score(y_val, predict)

    cm = metrics.confusion_matrix(y_val, predict)
    print(cm)


    xg_probs = xgb.predict_proba(X_val)[:, 1]
    roc_value = roc_auc_score(y_val, xg_probs)
    fpr, tpr, thresholds = roc_curve(y_val, xg_probs)

    print("Completed")
    return accuracy, precision, recall, f1

@PipelineDecorator.component(return_values=['accuracy','precision','recall','f1'], cache=True)
def rf_train():
    import pandas as pd
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score
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
    params = rf_model.get_params()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "RF_model.pkl")
    predict = rf_model.predict(X_val)

    accuracy = accuracy_score(y_val, predict)
    precision = precision_score(y_val, predict)
    recall = recall_score(y_val, predict)
    f1 = f1_score(y_val, predict)

    cm = metrics.confusion_matrix(y_val, predict)
    print(cm)

    rf_probs = rf_model.predict_proba(X_val)[:, 1]
    roc_value = roc_auc_score(y_val, rf_probs)
    fpr, tpr, thresholds = roc_curve(y_val, rf_probs)

    print("Completed")
    return accuracy, precision, recall, f1


@PipelineDecorator.component(return_values=['accuracy', 'precision', 'recall', 'f1'], cache=True)
def prediction():
    # make sure we have scikit-learn for this step, we need it to use to unpickle the object
    print(">>>>>>>>>>>Importing Modules>>>>>>>>>>>>>>>>>>>>>>>")
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score
    import joblib

    print(">>>>>>>>>>>>>Predicting..>>>>>>>>>>>>>>>>")
    with open("RF_model.pkl", 'rb') as f:
        model = joblib.load(f)
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

@PipelineDecorator.pipeline(name="Upsell_CrossSell_pipeline", project="examples", version="0.0.5")
def executing_pipeline(mock_parameter="mock"):
    print(mock_parameter)

    # Use the pipeline argument to start the pipeline and pass it ot the first step
    print("launch step one")
    accuracy, precision, recall, f1 = prediction()

if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('clearml-demo')
    PipelineDecorator.debug_pipeline()
    executing_pipeline()
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)

    print("process completed")