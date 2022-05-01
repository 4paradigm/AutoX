English | [简体中文](./README.md)
# what is autox_server?
AutoX supports online deployment services

# content
<!-- TOC -->

- [what is autox_server?](#autox_server是什么？)
- [Scope of application
](#适用范围)
- [Get started quickly](#快速上手)
- [Use case](#案例)
- [input format description](#输入格式说明)

<!-- /TOC -->

# Scope of application
- Currently AutoX supports binary classification scenarios
- Features to support regression and time series forecasting are under development

# Get started quickly
The structure of autox_server is divided into two parts: training and prediction.<br>
Step 1. Training: Obtain AutoX solutions (including data preprocessing, table collage, feature engineering, model selection, model tuning, model fusion, etc.) through data exploration of the training set, and save the solutions as pickles to in the specified path.
```
from autox import AutoXServer
# Specify the configuration file path
data_info_path = './bank/data_info.json'
# Specify the training data path
train_set_path = './bank/train_data'
# Initialize AutoXServer and execute AutoXServer training module, server_name customization
autoxserver = AutoXServer(is_train = True, server_name = 'bank',
                          data_info_path = data_info_path, train_set_path = train_set_path)
autoxserver.fit()
# Specify the path and save
autoxserver.save_server(save_path = './save_path')
```
Step 2. Prediction: Import the trained AutoXServer, call the prediction function, and get the prediction result.
```
from autox import AutoXServer
# Import the trained AutoXServer, note that the server_name needs to be the same as the training time
autoxserver = AutoXServer(is_train = False, server_name = 'bank')
autoxserver.load_server(save_path = './save_path')
# Specify the test set path, call the predict function of autoxserver, and return the prediction result (pandas dataframe format)
pred = autoxserver.predict(test_set_path = ./bank/test_data)
```

# Use case
### Case 1: Banking Scenario - Customer Loan Risk Prediction
Case description: Establish an accurate overdue prediction model to predict whether the user will overdue repayment through the user's basic information, consumption behavior, repayment situation, etc.<br>
Data download address：[Baidu network disk](https://pan.baidu.com/s/1OzxjH8a7qEhY0WYb5OjC2g)- extraction code:phgb, [google cloud](https://drive.google.com/file/d/1izyg93sN7F_Kb7K03rQFVRYt_952MoDq/view?usp=sharing)<br>
Detailed data description:[link](https://challenge.datacastle.cn/v3/cmptDetail.html?id=176) <br>
autox_server training code: [bank_train.ipynb](demo/bank/bank_train.ipynb)<br>
autox_server prediction code:[bank_test.ipynb](demo/bank/bank_test.ipynb)<br>


# input format description
An example of a data format that meets the requirements is as follows, including the data information file data_info.json, the training set folder, and the test set folder:
```
data_info.json
train_data/
    browse_train.csv
    bill_train.csv
    userinfo_train.csv
    bank_train.csv
    overdue_train.csv
test_data/
    browse_test.csv
    bill_test.csv
    userinfo_test.csv
    bank_test.csv
    overdue_test.csv
```

- data_info.json contains basic data information, table information and table relationships.
```
{
    "dataset_id": "Athena",  # data table name
    "recom_metrics": ["auc"], # Recommended evaluation metrics
    "target_entity": "overdue", #Main table (the table where the label column is located is the main table)
    "target_id": ["new_user_id"],  # id column
    "target_label": "label",  #  target value column
    "time_budget": 1200,  # time budget
    "entities": {  # Information about each table in the dataset
        "overdue": {  # overdue table
            "file_name": "overdue.csv",  # table name
            "format": "csv", # table format
            "header": "true", # does/does not contain header
            "is_static": "true", # whether it's static table
            "time_col": [], # The column name corresponding to the time column, only available for non-static tables
            "columns":  #  column data type
            [{"new_user_id": "Str"}, 
             {"label": "Num"},
             {"flag1": "Num"},
             {"mock_time": "Timestamp"},
             {"mock_labelEncoder": "Str"}]
        },
        "userinfo": {
            "file_name": "userinfo.csv",
            "format": "csv",
            "header": "true",
            "is_static": "true",
            "time_col": [],
            "columns": 
            [{"new_user_id": "Str"}, {"flag1": "Num"}, {"flag2": "Num"}, {"flag3": "Num"}, {"flag4": "Num"}, {"flag5": "Num"}]
        },
        "bank": {
            "file_name": "bank.csv",
            "format": "csv",
            "header": "true",
            "is_static": "false",
            "time_col": ["flag1"],
            "columns": 
            [{"new_user_id": "Str"}, {"flag1": "Num"}, {"flag2": "Num"}, {"flag3": "Num"}, {"flag4": "Num"}]
        },
        "browse": {
            "file_name": "browse.csv",
            "format": "csv",
            "header": "true",
            "is_static": "false",
            "time_col": ["flag1"],
            "columns": 
            [{"new_user_id": "Str"}, {"flag1": "Num"}, {"flag2": "Num"}, {"flag3": "Num"}]
        },
        "bill": {
            "file_name": "bill.csv",
            "format": "csv",
            "header": "true",
            "is_static": "false",
            "time_col": ["flag1"],
            "columns": 
            [{"new_user_id": "Str"}, {"flag1": "Num"}, {"flag2": "Num"}, {"flag3": "Num"}, {"flag4": "Num"}, {"flag5": "Num"}, {"flag6": "Num"}, {"flag7": "Num"}, {"flag8": "Num"}, {"flag9": "Num"}, {"flag10": "Num"}, {"flag11": "Num"}, {"flag12": "Num"}, {"flag13": "Num"}, {"flag14": "Num"}]
        }
    },
    "relations": [ # Table relationship (can be included as 1-1, 1-M, M-1, M-M four types)
        {
            "related_to_main_table": "true", # Whether it is related to the main table
            "left_entity": "overdue",  #  left table name
            "left_on": ["new_user_id"],  # Left table concat key
            "right_entity": "userinfo",  # right table name
            "right_on": ["new_user_id"], # right table concat table key
            "type": "1-1" # The connection relationship between the left table and the right table
        },
        {
            "related_to_main_table": "true",
            "left_entity": "overdue",
            "left_on": ["new_user_id"],
            "left_time_col": "flag1",
            "right_entity": "bank",
            "right_on": ["new_user_id"],
            "right_time_col": "flag1",
            "type": "1-M"
        },
        {
            "related_to_main_table": "true",
            "left_entity": "overdue",
            "left_on": ["new_user_id"],
            "left_time_col": "flag1",
            "right_entity": "browse",
            "right_on": ["new_user_id"],
            "right_time_col": "flag1",
            "type": "1-M"
        },
        {
            "related_to_main_table": "true",
            "left_entity": "overdue",
            "left_on": ["new_user_id"],
            "left_time_col": "flag1",
            "right_entity": "bill",
            "right_on": ["new_user_id"],
            "right_time_col": "flag1",
            "type": "1-M"
        }
    ]
}
```
