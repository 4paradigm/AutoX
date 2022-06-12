# AutoX Recommend Beta

## 核心模块

- Recall
- Feature Engineer
- Rank



## config.json

使用前，需要在config.json文件中进行相应配置，明确输入数据、pipeline选择等。

可以根据数据，从预设的pipeline选择合适方法；也可以自行组合构件相应pipeline。



| 字段 | 属性 | 说明 |
| ------------ | ---- | ---- |
| USERS        | userId | User数据表中User ID列名，必选项 |
|  | | |
| ITEMS        | itemId | Item数据表中Item ID列名，必选项 |
|  | | |
| INTERACTIONS | userId | Interaction表中User ID列名，必选项 |
|  | itemId | Item ID列名，必选项 |
|  | timestamp | Time列名，交互时间 |
|  |  |  |
| METHODS| recallNum | 召回Item数量 |
| | RECALL | 召回方法，目前支持 Popular、History、ItemCF、BinaryNet、W2V等方式，需要指定方法名name和所需要的数据表中的列名required_attrs，可设置方法参数parameters |
| | FEATURE_ENGINEER | 特征增强方法 |
| | RANK | 排序方法，目前支持LightGBM |



