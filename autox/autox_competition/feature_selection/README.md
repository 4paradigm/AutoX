# Adversarial Validation
### 说明
将训练集和测试集分布不一致的特征筛选出来并剔除。
原理：
```
第1步，准备好已完成特征构造的训练集和测试集；
第2步，删除训练集原始的label列，给训练集加标签列，值为1，给测试集加标签列，值为0；
第3步，将执行完第2步的训练集和测试集合并；
第4步，利用合并后的数据集，5折交叉训练lgb模型，计算5折的平均auc；
第5步，设置一定的阈值(默认阈值为0.6，阈值必须大于0.5)；
若平均auc大于阈值，记录特征重要性最高的特征，并将这个特征从合并的数据集中删除，并重新执行第4步；
若平均auc不大于阈值，进入第6步；
第6步，所有第5步中被删除的特征，就是最终获得的训练集和测试集分布不一致的特征，不推荐在后续过程中使用。
```
### 使用案例
- [Titanic_AdversarialValidation_AutoX](https://www.kaggle.com/code/poteman/titanic-adversarialvalidation-autox)

# GRN Feature Selection
### 说明
筛选重要性靠前的特征。
原理：
```
第1步，准备好包含至少一列数值类型为连续型的目标值的数据集，和对应的列定义，列定义格式如下：
column_definition = {
    'target':[目标值列名],
    'num':[连续型特征列1, 连续型特征列2, ..., 连续型特征列N],
    'cat':[离散型特征列1, 离散型特征列3, ..., 离散型特征列N]
}
第2步，根据列定义，取出N个num和N个cat列，使用MinMaxScaler对num列进行预处理后，划分为训练集和验证集，
处理为Dataloader输入到GRN和单层nn组成的模型中；
第3步，在模型中对cat输入进行embedding，并与num输入进行拼接，成为2*N的输入传给GRN；
第4步，GRN计算2*N个特征的权重，并将权重乘以特征输入，作为输出传给单层nn，映射到1维与target计算损失，根据损失反向更新权重；
第5步，模型每次迭代训练结束后在验证集计算一次损失，进行8次迭代训练后，取验证集上最优得分的特征权重作为最终结果；
第6步，根据所需的最终特征数量，选择权重中排名对应靠前的特征作为输出，并从原数据集中提取对应的特征列作为新的数据集。
```
### 使用案例
- [ubiquant_GRNFeatureSelection_AutoX](https://www.kaggle.com/hengwdai/grn-featureselection-autox)

