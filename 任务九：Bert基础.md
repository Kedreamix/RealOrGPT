# Coggle 30 Days of ML（23年7月）任务六：训练FastText、Word2Vec词向量



#### 任务九：学会Bert基础，transformer库基础使用

- 说明：在这个任务中，你将学习Bert模型的基础知识，并了解transformer库的基本使用方法，transformer库提供了Bert模型的实现。
- 实践步骤：
  1. 学习Bert模型的原理和架构。
  2. 了解transformer库的基本使用方法，包括Bert模型的初始化、输入编码和特征提取等操作。





## Bert模型

### 训练目标

BERT使用了维基百科等语料库数据，共几十GB，这是一个庞大的语料库。对于一个GB级的语料库，雇佣人力进行标注成本极高。BERT使用了两个巧妙方法来无监督地训练模型：Masked Language Modeling和Next Sentence Prediction。这两个方法可以无需花费时间和人力标注数据，以较低成本无监督地得到训练数据。图1就是一个输入输出样例。

对于Masked Language Modeling，给定一些输入句子（图1中最下面的输入层），BERT将输入句子中的一些单词盖住（图1中Masked层），经过中间的词向量和BERT层后，BERT的目标是让模型能够预测那些刚刚被盖住的词。还记得英语考试中，我们经常遇到“完形填空”题型吗？能把完形填空做对，说明已经理解了文章背后的语言逻辑。BERT的Masked Language Modeling本质上就是在做“完形填空”：预训练时，先将一部分词随机地盖住，经过模型的拟合，如果能够很好地预测那些盖住的词，模型就学到了文本的内在逻辑。

![img](http://aixingqiu-1258949597.cos.ap-beijing.myqcloud.com/2021-12-18-pretrain.png)图1 BERT预训练的输入和输出

除了“完形填空”，BERT还需要做Next Sentence Prediction任务：预测句子B是否为句子A的下一句。Next Sentence Prediction有点像英语考试中的“段落排序”题，只不过简化到只考虑两句话。如果模型无法正确地基于当前句子预测Next Sentence，而是生硬地把两个不相关的句子拼到一起，两个句子在语义上是毫不相关的，说明模型没有读懂文本背后的意思。





## HuggingFace Transformers

使用BERT和其他各类Transformer模型，绕不开[HuggingFaceopen in new window](https://huggingface.co/)提供的Transformers生态。HuggingFace提供了各类BERT的API（`transformers`库）、训练好的模型（HuggingFace Hub）还有数据集（`datasets`）。最初，HuggingFace用PyTorch实现了BERT，并提供了预训练的模型，后来。越来越多的人直接使用HuggingFace提供好的模型进行微调，将自己的模型共享到HuggingFace社区。HuggingFace的社区越来越庞大，不仅覆盖了PyTorch版，还提供TensorFlow版，主流的预训练模型都会提交到HuggingFace社区，供其他人使用。

使用`transformers`库进行微调，主要包括：

- Tokenizer：使用提供好的Tokenizer对原始文本处理，得到Token序列；
- 构建模型：在提供好的模型结构上，增加下游任务所需预测接口，构建所需模型；
- 微调：将Token序列送入构建的模型，进行训练。

### 安裝库

首先需要安装`transformer`库

```bash
pip install transformer
```

### BertTokenizer

BertTokenizer是用于Bert模型的分词器，它包含了Bert模型的词典。在进行词向量化之前，我们需要将文本映射到词典中的对应序号，以获取相应的词向量。

下面两行代码会创建 `BertTokenizer`，并将所需的词表加载进来。首次使用这个模型时，`transformers` 会帮我们将模型从HuggingFace Hub下载到本地。



```python
>>> from transformers import BertTokenizer
>>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
```

用得到的`tokenizer`进行分词：



```python
>>> encoded_input = tokenizer("我是一句话")
>>> print(encoded_input)
{'input_ids': [101, 2769, 3221, 671, 1368, 6413, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
```

得到的一个Python `dict`。其中，`input_ids`最容易理解，它表示的是句子中的每个Token在词表中的索引数字。词表（Vocabulary）是一个Token到索引数字的映射。可以使用`decode()`方法，将索引数字转换为Token。



```python
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] 我 是 一 句 话 [SEP]'
```

可以看到，`BertTokenizer`在给原始文本处理时，自动给文本加上了`[CLS]`和`[SEP]`这两个符号，分别对应在词表中的索引数字为101和102。`decode()`之后，也将这两个符号反向解析出来了。

`token_type_ids`主要用于句子对，比如下面的例子，两个句子通过`[SEP]`分割，0表示Token对应的`input_ids`属于第一个句子，1表示Token对应的`input_ids`属于第二个句子。不是所有的模型和场景都用得上`token_type_ids`。



```python
>>> encoded_input = tokenizer("您贵姓?", "免贵姓李")
>>> print(encoded_input)
{'input_ids': [101, 2644, 6586, 1998, 136, 102, 1048, 6586, 1998, 3330, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

句子通常是变长的，多个句子组成一个Batch时，`attention_mask`就起了至关重要的作用。



```python
>>> batch_sentences = ["我是一句话", "我是另一句话", "我是最后一句话"]
>>> batch = tokenizer(batch_sentences, padding=True, return_tensors="pt")
>>> print(batch)
{'input_ids': 
 tensor([[ 101, 2769, 3221,  671, 1368, 6413,  102,    0,    0],
        [ 101, 2769, 3221, 1369,  671, 1368, 6413,  102,    0],
        [ 101, 2769, 3221, 3297, 1400,  671, 1368, 6413,  102]]), 
 'token_type_ids': 
 tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': 
 tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

对于这种`batch_size = 3`的场景，不同句子的长度是不同的，`padding=True`表示短句子的结尾会被填充`[PAD]`符号，`return_tensors="pt"`表示返回PyTorch格式的`Tensor`。`attention_mask`告诉模型，哪些Token需要被模型关注而加入到模型训练中，哪些Token是被填充进去的无意义的符号，模型无需关注



### BertModel

下面两行代码会创建`BertModel`，并将所需的模型参数加载进来。

```python
>>> from transformers import BertModel
>>> model = BertModel.from_pretrained("bert-base-chinese")
```

`BertModel`是一个PyTorch中用来包裹网络结构的`torch.nn.Module`，`BertModel`里有`forward()`方法，`forward()`方法中实现了将Token转化为词向量，再将词向量进行多层的Transformer Encoder的复杂变换。

`forward()`方法的入参有`input_ids`、`attention_mask`、`token_type_ids`等等，这些参数基本上是刚才Tokenizer部分的输出。

```python
>>> bert_output = model(input_ids=batch['input_ids'])
```

`forward()`方法返回模型预测的结果，返回结果是一个`tuple(torch.FloatTensor)`，即多个`Tensor`组成的`tuple`。`tuple`默认返回两个重要的`Tensor`：

```python
>>> len(bert_output)
2
```

- **last_hidden_state**：输出序列每个位置的语义向量，形状为：(batch_size, sequence_length, hidden_size)。
- **pooler_output**：`[CLS]`符号对应的语义向量，经过了全连接层和tanh激活；该向量可用于下游分类任务。





### 下游任务

BERT可以进行很多下游任务，`transformers`库中实现了一些下游任务，我们也可以参考`transformers`中的实现，来做自己想做的任务。比如单文本分类，`transformers`库提供了`BertForSequenceClassification`类。

```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = ...
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        ...
        
    def forward(
        ...
    ):
        ...

        outputs = self.bert(...)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        ...
```

在这段代码中，`BertForSequenceClassification`在`BertModel`基础上，增加了`nn.Dropout`和`nn.Linear`层，在预测时，将`BertModel`的输出放入`nn.Linear`，完成一个分类任务。除了`BertForSequenceClassification`，还有`BertForQuestionAnswering`用于问答，`BertForTokenClassification`用于序列标注，比如命名实体识别。

`transformers` 中的各个API还有很多其他参数设置，比如得到每一层Transformer Encoder的输出等等，可以访问他们的[文档open in new window](https://huggingface.co/docs/transformers/)查看使用方法。

### 一词多义

```python
import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
input_ids = torch.tensor(tokenizer.encode("good night")).unsqueeze(0)  # Batch size 1
input_ids2 = torch.tensor(tokenizer.encode("good food")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
outputs2 = model(input_ids2)
print(outputs[0][0][1]) # 取出good night 中的 good
print(outputs2[0][0][1]) # 取出good food 中的 good
```

看的出来，同一个词，在不同的句子中词向量是不同的。
因此bert能够很好的解决一次多义的现象，这便是它的魅力所在



参考

- [基于transformers的自然语言处理(NLP)入门](https://datawhalechina.github.io/learn-nlp-with-transformers/)
- [BERT](https://lulaoshi.info/deep-learning/attention/bert.html)
