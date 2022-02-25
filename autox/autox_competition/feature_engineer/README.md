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
