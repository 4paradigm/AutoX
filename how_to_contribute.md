# How to contribute

我们希望将AutoX打造成为一款性能、效果和稳定性均达到国际领先水平的自动机器学习解决方案。为了实现这一愿景，我们非常欢迎开发者们贡献一份力量，也将相应地给于贡献者激励以表认可和感谢。

在贡献代码、文档或Issue之前，请阅读以下指引。

## 联系方式

- AutoX贡献者意向成员群
  <img src="./img/developers_0531.jpeg" width = "200" height = "260" alt="wechat" align=center/>
- Email: caihengxing@4paradigm.com

## AutoX可贡献内容

### 内容完善

- 发现代码bug 提交Issue
- 修复bug, 提交补丁代码
- 撰写和改进项目文档(中英文wiki)
- AutoX网页优化(sphinx编写)
- 添加使用案例Demo [参考](https://github.com/4paradigm/AutoX/blob/master/demo/stumbleupon/kaggle_stumbleupon_autox.ipynb)
- 添加AutoX在Benchmark数据集的测试性能效果(与h2o和autogluon对比)[参考](https://github.com/4paradigm/AutoX/tree/master/demo/stumbleupon)
- 对目前AutoX的代码中的效率、格式、注释等进行优化
- 参与Issue的讨论，如答疑解惑、提供想法或报告无法解决的错误

### 功能开发

#### 功能性api贡献

可选贡献方向包括：数据预处理方法、拼表技术或副表特征、特征工程、特征选择方法、自动调参方法、模型融合技术、metric设计

##### 功能性api代码要求说明: 

1. api的接口建议仿照sklearn的接口设计, 例如类需要实现一个fit和transform函数;

2. 给出该api的简要说明;

3. 在一个公开数据集上执行该功能的案例，给出案例公开链接.建议使用kaggle数据集以及kernel，并将kernel public.

   功能性api贡献参考示例: [gbdt特征代码](https://github.com/4paradigm/AutoX/blob/master/autox/autox_competition/feature_engineer/fe_gbdt.py), [gbdt特征案例](https://www.kaggle.com/code/poteman/ubiquant-gbdt-features/notebook?scriptVersionId=88706805)

#### automl pipeline贡献

1. 给出端到端完整的pipeline代码;

2. 给出pipeline设计逻辑架构图;

3. 给出在不同数据场景下和autogluon和h2o的效果对比.

   automl pipeline贡献参考示例:[参考](https://github.com/4paradigm/AutoX/blob/master/autox/autox.py)

### 其它规划方向的开发

- 多分类任务
- 时空预测任务
- 多模态任务
- 分布式版本AutoX

### 社区运营

- issues解答
- 社区宣传
- 数据竞赛baseline编写

## 你将获得

参与开源社区，你将获得技术能力提升、自身声望积累、人际关系拓展以及个人综合素质提升, 包括沟通协作、解决问题的能力等。除此之外，

## 优秀的贡献者有机会获得

- 科研合作机会: 依托autox项目，和第四范式研究员以及高校顾问合作开展科研项目
- 第四范式正式员工/实习面试直通车

## 贡献方式

建议通过pull request的方式提交修改，对git不熟悉的朋友可以参考这个[链接](https://gitbeijing.com/fork_flow.html)。

我们的代码团队会监控pull request, 进行相应的代码测试和检查，通过的pr会被合并至Master分支。
