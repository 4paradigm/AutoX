# Count特征
### 说明
将类别型特征转化为其出现的次数。
举例：将商品id在转化成商品id在全量数据集中出现的次数。

# Cumsum特征
### 说明
对某一列进行聚合，求另一列的累计求和。
举例：每一条消费记录为一条样本，对用户聚合，求消费金额的cumsum。表示该笔消费之后，用户累计消费金额。

# Denoising_autoencoder特征
### 说明
todo

# Diff特征
### 说明
按时间排序后，对某一列进行聚合，求另一列当前样本和前N条(或后N条)样本的差值。
举例：按时间排序后，计算用户当前笔消费消费金额和上一笔消费消费金额的差值。

# Dimension_reduction特征
### 说明
用pca、ica、grp、srp四种降维方法对原始特征进行降维，将降维之后的结果作为特征
### 调用方式
```
from autox.autox_competition.feature_engineer import FeatureDimensionReduction
featureDimensionReduction = FeatureDimensionReduction()
featureDimensionReduction.fit(df, id_column = ['row_id','time_id','investment_id'], target = 'target')
dr_feature = featureDimensionReduction.transform(df)
```
### 使用案例
- [kaggle_ubiquant_DimensionReduction_Feature](https://www.kaggle.com/poteman/ubiquant-dimensionreduction-feature/notebook)


# Exp_weighted_mean特征
### 说明
指数移动平均值

# GBDT特征
### 说明
将样本输入到训练好的gbdt模型(例如一个包含30颗树的gbdt模型)中，将样本落入到每棵树的叶子结点的编号作为特征。
### 调用方式
```
from autox.autox_competition.feature_engineer import FeatureGbdt
featureGbdt = FeatureGbdt()
featureGbdt.fit(X_train, y_train, objective= 'binary', num_of_features = 50)
lgb_feature_train = featureGbdt.transform(X_train)
lgb_feature_test = featureGbdt.transform(X_test)
```
### 使用案例
- [kaggle_Ubiquant_Market_Prediction](https://www.kaggle.com/poteman/ubiquant-gbdt-features?scriptVersionId=88706805)

# Image2vec特征
### 说明
将图片输入转化为向量特征。

# Nlp特征
### 说明
对于识别为长文本的列，提取nlp信息。

# rank特征
### 说明
对某一列进行聚合，求另一列在聚合窗口内的排序值。
举例：计算当前样本属于用户在当天内第几次出现的样本

# Rolling_stat_ts特征
### 说明
时序类特征，计算滚动窗口内的统计特征(均值、方差、中位数、最大值、最小值)。

# Shift特征
### 说明
对某一列聚合，获得另一列在前N条(或后N条)样本中的值。
举例：获得用户在上一条记录中的违约情况。

# Shift_ts特征
时序类特征，获得lag信息。

# Stat特征
对某一列聚合，获得另一列在窗口内的统计信息(对于连续型变量求均值、最小值、最大值、中位数、方差，
对于类别型变量求nunique)

# Target_encoding特征
将类别型变量转化为对应类别下标签的平均值。
举例：标签为年收入，将学历(类别型变量)转化为对应学历的平均年收入。

# Time特征
将时间列特征进行分解。
获得信息包括：年、月、日、时、一年的第几周、星期、是否工作日、季度、是否月初、是否月末。