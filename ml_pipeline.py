from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'Saksham Arora',
    'depends_on_past': False,
    'start_date': days_ago(31),
    'email': ['example@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

scripts_path=Variable.get("data_path")

#instantiates a directed acyclic graph
dag = DAG(
    'dev-model-analyzing-daily',
    default_args=default_args,
    description='A Machine Learning pipeline analyzing multiple models.',
    schedule_interval='@daily',
    template_searchpath=scripts_path
)



def data_read():
    df = pd.read_csv(f"{scripts_path}general_data.csv")
    df.head()

def data_clean():

    df = pd.read_csv(f"{scripts_path}general_data.csv")
    print(df.head())
    df.info() #checking information about the dataset

    lst=df.columns[df.isna().any()].tolist()
    for i in lst:
        df[i]=df[i].replace(np.nan,df[i].median())

    df.drop(['EmployeeCount','Over18','StandardHours','StockOptionLevel'],axis=1,inplace=True)
    print(df.columns)

def data_quality_check():
    df = pd.read_csv(f"{scripts_path}general_data.csv")
    df.head()
    df.info() #checking information about the dataset

    lst=df.columns[df.isna().any()].tolist()
    for i in lst:
        df[i]=df[i].replace(np.nan,df[i].median())

    df.drop(['EmployeeCount','Over18','StandardHours','StockOptionLevel'],axis=1,inplace=True)
    df.columns

    from sklearn.preprocessing import LabelEncoder
    label_encoder_y=LabelEncoder()

    '''Identifying the columns with string'''
    df_num = df.select_dtypes(exclude=[np.number])
    lst=list(df_num.columns)

    '''Converting those columns to integer values'''
    for i in lst:
        df[i]=label_encoder_y.fit_transform(df[i])
    df.head()

# instantiate tasks using Operators.
#BashOperator defines tasks that execute bash scripts. In this case, we run Python scripts for each task.

def lr_model(**kwargs):
    df = pd.read_csv(f"{scripts_path}general_data.csv")
    df.head()
    df.info() #checking information about the dataset

    lst=df.columns[df.isna().any()].tolist()
    for i in lst:
        df[i]=df[i].replace(np.nan,df[i].median())

    df.drop(['EmployeeCount','Over18','StandardHours','StockOptionLevel'],axis=1,inplace=True)
    df.columns

    from sklearn.preprocessing import LabelEncoder
    label_encoder_y=LabelEncoder()

    '''Identifying the columns with string'''
    df_num = df.select_dtypes(exclude=[np.number])
    lst=list(df_num.columns)

    '''Converting those columns to integer values'''
    for i in lst:
        df[i]=label_encoder_y.fit_transform(df[i])
    df.head()



    y = df['Attrition']
    X= df.drop('Attrition', axis = 1)

    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state=44)
    from sklearn.preprocessing import StandardScaler
    Scaler_X = StandardScaler()
    X_train = Scaler_X.fit_transform(X_train)
    X_test = Scaler_X.transform(X_test)

    from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(solver='liblinear', random_state = 44)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)

    print("The accuracy score is: ", format(accuracy_score(y_test,y_pred),".2f"))
    print(" ")
    print("Confusion matrix: ","\n",confusion_matrix(y_test,y_pred))

    return accuracy_score(y_test,y_pred)

def rf_model(**kwargs):
    df = pd.read_csv(f"{scripts_path}general_data.csv")
    df.head()
    df.info() #checking information about the dataset

    lst=df.columns[df.isna().any()].tolist()
    for i in lst:
        df[i]=df[i].replace(np.nan,df[i].median())

    df.drop(['EmployeeCount','Over18','StandardHours','StockOptionLevel'],axis=1,inplace=True)
    df.columns

    from sklearn.preprocessing import LabelEncoder
    label_encoder_y=LabelEncoder()

    '''Identifying the columns with string'''
    df_num = df.select_dtypes(exclude=[np.number])
    lst=list(df_num.columns)

    '''Converting those columns to integer values'''
    for i in lst:
        df[i]=label_encoder_y.fit_transform(df[i])
    df.head()

    y = df['Attrition']
    X= df.drop('Attrition', axis = 1)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state=44)
    from sklearn.preprocessing import StandardScaler
    Scaler_X = StandardScaler()
    X_train = Scaler_X.fit_transform(X_train)

    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    RF_model = RandomForestClassifier(n_estimators=50,max_depth=3,random_state=0)
    RF_model.fit(X_train, y_train)
    y_predict_rf = RF_model.predict(X_test)
    RF_model_score = (format(RF_model.score(X_test, y_test),".2f"))
    print("The accuracy score using Random Forest Classifier is: ",RF_model_score)
    print(classification_report(y_test,y_predict_rf))

    return RF_model_score

def dt_model(**kwargs):
    df = pd.read_csv(f"{scripts_path}general_data.csv")
    df.head()
    df.info() #checking information about the dataset

    lst=df.columns[df.isna().any()].tolist()
    for i in lst:
        df[i]=df[i].replace(np.nan,df[i].median())

    df.drop(['EmployeeCount','Over18','StandardHours','StockOptionLevel'],axis=1,inplace=True)
    df.columns

    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
    label_encoder_y=LabelEncoder()

    '''Identifying the columns with string'''
    df_num = df.select_dtypes(exclude=[np.number])
    lst=list(df_num.columns)

    '''Converting those columns to integer values'''
    for i in lst:
        df[i]=label_encoder_y.fit_transform(df[i])
    df.head()

    y = df['Attrition']
    X= df.drop('Attrition', axis = 1)

    from sklearn import tree
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state=44)
    from sklearn.preprocessing import StandardScaler
    Scaler_X = StandardScaler()
    X_train = Scaler_X.fit_transform(X_train)
    dec_tree_model = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
    dec_tree_model.fit(X_train, y_train)
    y_predict = dec_tree_model.predict(X_test)
    dec_tree_model_score = format(dec_tree_model.score(X_test, y_test))
    

    print("The accuracy score using Decision Tree Classifier is: ",dec_tree_model_score)

    print(classification_report(y_test,y_predict))
    return dec_tree_model_score

def best_model(**kwargs):
    ti = kwargs['ti']
    lr_score = ti.xcom_pull(task_ids='logistic_regression_model')
    dt_score = ti.xcom_pull(task_ids='decision_tree_classifier')
    rf_score =  ti.xcom_pull(task_ids='random_forest_classifier')
    # print(type(lr_score),lr_score)
    # print(type(dt_score),dt_score)
    # print(type(rf_score),rf_score)
    # print(type(float(lr_score)),lr_score)
    # print(type(float(dt_score)),dt_score)
    # print(type(float(rf_score)),rf_score)
    print(max(float(lr_score),float(dt_score),float(rf_score)))



data_read = PythonOperator(
    task_id='data_read',
    python_callable=data_read,
    dag=dag,
)
data_clean = PythonOperator(
    task_id='data_cleansing',
    depends_on_past=False,
    python_callable=data_clean,
    retries=3,
    dag=dag,
)
serve_commands = """
    lsof -i tcp:8008 | awk 'NR!=1 {print $2}' | xargs kill;
    python3 /usr/local/airflow/scripts/serve.py serve
    """
data_quality_check = PythonOperator(
    task_id='data_quality_check',
    depends_on_past=False,
    python_callable=data_quality_check,
    retries=3,
    dag=dag,
)

lr_model = PythonOperator(
    task_id='logistic_regression_model',
    depends_on_past=False,
    python_callable=lr_model,
    retries=3,
    provide_context=True,
    dag=dag,
)

rf_model = PythonOperator(
    task_id='random_forest_classifier',
    depends_on_past=False,
    python_callable=rf_model,
    provide_context=True,
    retries=3,
    dag=dag,
)

dt_model = PythonOperator(
    task_id='decision_tree_classifier',
    depends_on_past=False,
    python_callable=dt_model,
    provide_context=True,
    retries=3,
    dag=dag,
)


best_model=PythonOperator(
    task_id='best_model',
    depends_on_past=False,
    python_callable=best_model,
    provide_context=True,
    retries=3,
    dag=dag,
)

#sets the ordering of the DAG. The >> directs the 2nd task to run after the 1st task. This means that
#download images runs first, then train, then serve.
data_read >> data_clean >> data_quality_check
data_quality_check >> lr_model
data_quality_check >> rf_model
data_quality_check >> dt_model
lr_model >> best_model
rf_model >> best_model
dt_model >> best_model
