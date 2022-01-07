# autox_server是什么？
AutoX支持上线部署的服务

# 目录
<!-- TOC -->

- [autox_server是什么？](#autox_server是什么？)
- [适用范围](#适用范围)
- [快速上手](#快速上手)
- [案例](#案例)
- [输入格式说明](#输入格式说明)

<!-- /TOC -->

# 适用范围
- 当前AutoX支持二分类场景
- 支持回归及时序预测的功能正在开发中

# 快速上手
autox_server在结构上分训练和预测两部分。<br>
Step 1. 训练：通过对训练集进行数据探索，获得AutoX的解决方案(包含数据预处理、拼表、特征工程、模型选择、
模型调参、模型融合等)，将解决方案以pickle各保存到指定路径中。
```
from autox import AutoXServer
# 指定配置文件路径
data_info_path = './bank/data_info.json'
# 指定训练数据路径
train_set_path = './bank/train_data'
# 初始化AutoXServer并执行AutoXServer训练模块, server_name自定义
autoxserver = AutoXServer(is_train = True, server_name = 'bank',
                          data_info_path = data_info_path, train_set_path = train_set_path)
autoxserver.fit()
# 指定路径并保存
autoxserver.save_server(save_path = './save_path')
```
Step 2. 预测：将导入训练好的AutoXServer，调用预测函数，获得预测结果。
```
from autox import AutoXServer
# 导入训练好的AutoXServer
autoxserver = AutoXServer(is_train = False, server_name = 'bank')
autoxserver.load_server(save_path = './save_path')
# 指定测试集路径，调用autoxserver的predict函数，返回预测结果(pandas dataframe格式)
pred = autoxserver.predict(test_set_path = ./bank/test_data)
```

# 案例
### 案例1：银行场景-客户贷款风险预测
案例描述：通过用户基本信息，消费行为，还款情况等，建立准确的逾期预测模型，以预测用户是否会逾期还款。<br>
数据下载地址：[百度网盘](), [google cloud]()<br>
详细数据说明：[link](https://challenge.datacastle.cn/v3/cmptDetail.html?id=176) <br>
autox_server训练代码：[bank_train.ipynb](demo/bank/bank_train.ipynb)<br>
autox_server预测代码：[bank_test.ipynb](demo/bank/bank_train.ipynb)<br>


# 输入格式说明
一个符合要求的数据格式例子如下, 包括数据信息文件data_info.json、训练集文件夹、测试集文件夹：
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

- data_info.json包含数据基础信息、各表信息和表关系。
```
{
    "dataset_id": "Athena",  # 数据表名称 
    "recom_metrics": ["auc"], # 推荐的评价指标
    "target_entity": "overdue", #主表(label列所在表为主表)
    "target_id": [],  # id列
    "target_label": "label",  # 目标值列
    "time_budget": 1200,  # 时间预算
    "entities": {  # 数据集中各表的信息
        "overdue": {  # overdue表
            "file_name": "overdue.csv",  # 表名
            "format": "csv", # 表格式
            "header": "true", # 是否有header
            "is_static": "true", # 是否是静态表
            "time_col": [], # 时间列对应的列名，只有非静态表才有
            "columns":  # 列的数据类型
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
    "relations": [ # 表关系(可以包含为1-1, 1-M, M-1, M-M四种)
        {
            "related_to_main_table": "true", # 是否为和主表的关系
            "left_entity": "overdue",  # 左表名字
            "left_on": ["new_user_id"],  # 左表拼表键
            "right_entity": "userinfo",  # 右表名字
            "right_on": ["new_user_id"], # 右表拼表键
            "type": "1-1" # 左表与右表的连接关系
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
