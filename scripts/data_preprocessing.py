import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from clearml import Task, Dataset
import warnings
warnings.filterwarnings('ignore')

task = Task.init(project_name='UpSell_CrossSell_project', task_name='data_preprocessing', task_type="data_processing")

dataset =  Dataset.create(dataset_name = "Raw_Data",
                          dataset_prject = "UpSell_CrossSell_project",
                          dataset_version = 1.0,
                          description = "raw data")
dataset.add_files(path = 'dataset/Raw_Data.csv')
dataset.upload()
dataset.finalize()

print("Preprocessing")
train = pd.read_csv('dataset/Raw_Data.csv')

print(train.head())
print(train["Response"].value_counts())
print(train.shape)

d = []
for i in train['Age']:
    if i>=20 and i<=32:
        d.append('20-32')
    elif i>=33 and i<=52:
        d.append('33-52')
    elif i>=53 and i<=65:
        d.append('53-65')
    else:
        d.append('>65')

u = pd.DataFrame(d,columns=['binn_age'])
train = pd.concat([u,train],axis=1)
train.drop(["Age"], axis=1, inplace=True)
print(train.shape)

num_cols= ['Region_Code', 'Annual_Premium', 'Vintage']

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
    print(f"{col} : {check_outlier(train,col)}")

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(train, col)

# train.corrwith(train["Response"]).sort_values(ascending=False)
corr_df = train.corr()
plt.figure(figsize=(12, 9))
plot = sns.heatmap(corr_df, annot=True, xticklabels=corr_df.columns, yticklabels=corr_df.columns)
plt.show()
task.get_logger().report_matplotlib_figure("Correlation Matrix",'UpSell_CrossSell_project',plot,report_image=False,report_interactive=True)

# corr_df = corr_df.corr().unstack().sort_values().drop_duplicates()
# corr_df = pd.DataFrame(corr_df, columns=["corr"])
# corr_df.index.names = ['1', '2']
# corr_df = corr_df.reset_index()
# corr_df.sort_values(by="corr", ascending=True).head(30)

# high_corr = corr_df[(corr_df["corr"] >= 0.70) | (corr_df["corr"] <= -0.70)]
# high_corr

plot = sns.countplot(train.Gender)
plt.show()
task.get_logger().report_matplotlib_figure("Gender and Response",'UpSell_CrossSell_project',plot,report_image=False,report_interactive=True)

df = train.groupby(['Vehicle_Age','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
plot = sns.catplot(x="Vehicle_Age", y="count",col="Response",data=df, kind="bar",height=4, aspect=.7)
task.get_logger().report_matplotlib_figure("Vehicle Age and Response",'UpSell_CrossSell_project',plot,report_image=False,report_interactive=True)

df = train.groupby(['Vehicle_Damage','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
plot = sns.catplot(x="Vehicle_Damage", y="count",col="Response", data=df, kind="bar",height=4, aspect=.7)
task.get_logger().report_matplotlib_figure("Vehicle Damage and Response",'UpSell_CrossSell_project',plot,report_image=False,report_interactive=True)

train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)

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
    res = int(reg_res[i]/region.values[i]*100)
    percent.append(res)

region_code = pd.concat([region.reset_index(), pd.DataFrame(reg_res,columns=['reg_response']), pd.DataFrame(percent,columns=['percentage'])],axis=1)
region_code.columns = ['Region_Code', 'sum', 'reg_response', 'percentage']

reg_code = []
for i in train['Region_Code']:
    res = region_code[region_code['Region_Code']==i]['percentage'].values[0]
    if res<=5:
        reg_code.append('less than 5')  # A class
    elif res>5 and res<=10:
        reg_code.append('5-10')
    elif res>10 and res<=15:
        reg_code.append('10-15')
    else:
        reg_code.append('greter than 15')

train = pd.concat([pd.DataFrame(reg_code,columns=['binn_region_code']),train],axis=1)
train = train.drop('id',axis=1)

print(train.head())

psc = train['Policy_Sales_Channel'].value_counts()
print(psc)
psc_res = []
for i in psc.index:
    res = train[train['Policy_Sales_Channel']==i]['Response'].value_counts()
    if 1 in res.index:
        psc_res.append(res[1])
    else:
        psc_res.append(0)

per = []
for i in range(len(psc_res)):
    res = int(psc_res[i]/psc.values[i]*100)
    per.append(res)

policy_chh = pd.concat([psc.reset_index(),pd.DataFrame(psc_res,columns=['psc_res']),pd.DataFrame(per,columns=['Percentage'])],axis=1)

pol_sal_chh = []
for i in train['Policy_Sales_Channel']:
    res = policy_chh[policy_chh['index']==i]['Percentage'].values[0]
    if res<=5:
        pol_sal_chh.append('less than 5')
    elif res>5 and res<=10:
        pol_sal_chh.append('5-10')
    elif res>10 and res<=15:
        pol_sal_chh.append('10-15')
    elif res>15 and res<=20:
        pol_sal_chh.append('15-20')
    elif res>20 and res<=25:
        pol_sal_chh.append('20-25')
    else:
        pol_sal_chh.append('greater than 25')

train = pd.concat([pd.DataFrame(pol_sal_chh,columns=['binn_policy_sales_channel']),train],axis=1)
print(train.head())

v = []
for i in train['Vintage']:
    if i<=50:
        v.append('<=50')
    elif i>=51 and i<=100:
        v.append('51-100')
    elif i>=101 and i<=150:
        v.append('101-150')
    elif i>=151 and i<=200:
        v.append('151-200')
    elif i>=201 and i<=250:
        v.append('201-250')
    else:
        v.append('>250')

vintage = pd.DataFrame(v,columns=['binn_vintage'])
train = pd.concat([vintage, train],axis=1)
train = train.drop(columns=['Region_Code','Policy_Sales_Channel', 'Vintage'])

print(train.head())

train['binn_vintage'] = train['binn_vintage'].map({'51-100': "A", '101-150': "B", "151-200":"C","201-250":"D",">250" :"E", "<=50":"F"})
train['binn_policy_sales_channel'] = train['binn_policy_sales_channel'].map({'15-20': "A", 'less than 5': "B", "20-25 ":"C","greater than 25":"D","10-15" :"E", "5-10":"F"})
train['binn_region_code'] = train['binn_region_code'].map({'5-10': "A", 'greter than 15': "B", "10-15":"C","less than 5":"D"})
train['Vehicle_Age'] = train['Vehicle_Age'].map({'1-2 Year': "A", "< 1 Year": "B", "> 2 Years":"C"})

# train_1= train.groupby("binn_policy_sales_channel")["Response"].sum()
# plot = train_1.plot.pie(autopct="%.1f%%",figsize=(7,7))
# plt.show()
# task.get_logger().report_matplotlib_figure("binn_policy_sales_channel distribution",'UpSell_CrossSell_project',plot,report_image=False,report_interactive=True)

new_df = pd.get_dummies(train,drop_first=True)

dataset =  Dataset.create(dataset_name = "Preprocessed_Data",
                          dataset_prject = "UpSell_CrossSell_project",
                          dataset_version = 2.0,
                          description = "preprocessed data")
dataset.add_files(path = 'dataset/Processed_Data.csv')
dataset.upload()
dataset.finalize()

print("Completed")
task.close()
