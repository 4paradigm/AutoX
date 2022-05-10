[English](./README_EN.md) | 简体中文

# autox_recommend是什么

- 问题定义: next-item prediction, 预测用户未来一段时间点击/购买的商品

# 快速上手

# 目录
<!-- TOC -->

- [autox_recommend是什么是什么](#autox_recommend是什么是什么)
- [快速上手](#快速上手)
- [目录](#目录)
- [框架](#框架)
- [效果对比](#效果对比)

<!-- /TOC -->

# 框架
### 数据集
[数据集链接](./datasets/README.md)

### 召回模型
- 流行项目召回
- 历史购买召回
- ItemCF召回
- UserCF召回
- BinaryNet召回
- cate pop召回
- content based召回(图片, 文本描述等)

### 排序模型

#### 特征工程
- 交互特征
- 用户特征
- 商品特征

#### 模型
- lgb ranker(对于每一个用户, 对其候选的商品集进行排序)



# 效果对比
