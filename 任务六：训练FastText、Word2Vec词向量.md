# Coggle 30 Days of ML（23年7月）任务六：训练FastText、Word2Vec词向量



#### 任务六：学会训练FastText、Word2Vec词向量

- 说明：在这个任务中，你将学习如何训练FastText和Word2Vec词向量模型，这些词向量模型可以捕捉文本中的语义信息。
- 实践步骤：
  1. 准备大规模文本语料库。
  2. 使用FastText或gensim库中的Word2Vec类，设置相应的参数（如词向量维度、窗口大小、训练迭代次数等）来构建词向量模型。
  3. 使用Word2Vec类的build_vocab()方法，构建词汇表。
  4. 使用Word2Vec类的train()方法，训练词向量模型。





## Word2Vec

word2vec主要包括Skip-Gram和CBOW两种模型。

Skip-Gram是利用一个词语作为输入，来预测它周围的上下文 ，CBOW是利用一个词语的上下文作为输入，来预测这个词语本身 。

> word2vec模型其中有两种训练方式：skip-gram与CBOW，此外，还有两种加速训练的trick：hierarchical sofmtmax与negative sampling。

### 简单情形

输入：one-hot encoder

隐层：对输入的词求和取平均+线性层

输出：隐层

**word2vec的本质：**是一种**降维**操作——把词语从 one-hot encoder 形式的表示降维到 Word2vec 形式的表示，是一个|V|分类问题。

**训练目标：** 极小化负对数似然

**Tip：** CBOW的窗口内部丢掉了词序信息，但是在窗口滑动过程中是按照通顺的自然语言顺序滑动的，或者样本就是按照一定的语序取出来的，所以最终得出的词向量中会包含一定的语序信息。

### Word2vec 的训练trick

Word2vec 本质上是一个语言模型，它的输出节点数是 V 个，对应了 V 个词语，本质上是一个多分类问题，但实际当中，词语的个数非常非常多，会给计算造成很大困难，所以需要用技巧来加速训练。

- **Word pairs and “phase”**: 对常见的单词进行组合或者词组作为单个words来处理（比如`"New York"`,`"United Stated"`）

- **高频词采样：**对高频词进行抽样来减少训练样本的个数（比如`the`这个词在很多次的窗口中出现吗，他对其他词语义学习的帮助不大，并且，更多包含`the`的训练数据远远超出了学习`the`这个词向量所需要的样本数）

- **负采样：**用来提高训练速度并且改善所得到词向量的质量 。

  随机选取部分词作为负样本（比如当vocab_size为10000时，训练样本 `input word:"fox",output word:"quick"` ,在9999个负样本的权重中选择5-20个更新，参数大大减少）。

  - 如何选择negative words？

    根据词频进行负采样，出现概率高的单词容易被选为负样本。 每个单词被赋予一个权重。

  - **层次softmax（Hierarchical Softmax）**

    在进行最优化的求解过程中：从隐藏层到输出的Softmax层的计算量很大，因为要计算所有词的Softmax概率，再去找概率最大的值。

    word2vec采用了**霍夫曼树**来代替从隐藏层到输出softmax层的映射 。

    和之前的神经网络语言模型相比，霍夫曼树的所有内部节点就类似之前神经网络隐藏层的神经元,其中，根节点的词向量对应我们的投影后的词向量，而所有叶子节点就类似于之前神经网络softmax输出层的神经元，叶子节点的个数就是词汇表的大小。在霍夫曼树中，隐藏层到输出层的softmax映射不是一下子完成的，而是沿着霍夫曼树一步步完成的，因此这种softmax取名为 **Hierarchical Softmax** 。

    在word2vec中，我们采用了二元逻辑回归的方法，即规定沿着左子树走，那么就是负类(霍夫曼树编码1)，沿 着右子树走，那么就是正类(霍夫曼树编码0)。判别正类和负类的方法是使用sigmoid函数， 采用随机梯度上升求解二分类，每计算一个样本更新一次误差函数 。

    **使用霍夫曼树有什么好处：**

 首先，由于是二叉树，之前计算量为`V`,现在变成了`log2V`。

 第二，由于使用霍夫曼树是**高频的词靠近树根**，这样高频词需要更少的时间会被找到，这符合贪心优化思想。

## fasttext

fasttext是基于浅层神经网络训练的，其训练方式与word2vec中的CBOW方式如出一辙，fasttext是对整个句子的n-gram特征相加求平均，得到句向量，在根据句向量做分类。

fasttext的输入：embedding过的单词的词向量和n-gram向量

内存考虑：**哈希映射**，将n-gram映射到固定K个的索引上，相同的索引共享相同的embedding。



## fasttext与word2vec对比

- fasttext用作分类是有监督的，word2vec是无监督的
- fasttext输入部分考虑了n-gram特征，word2vec的输入只有one-hot encoder
- fasttext可以表示oov的单词

word2vec的不足：

1. 多义词的问题。
2. Word2vec 是一种静态的方式，无法针对特定任务做动态优化

fasttext的不足：

fastText很难学出词序对句子语义的影响，对复杂任务还是需要用复杂网络学习任务的语义表达。



# gensim中word2vec的使用

首先我们先介绍一下gensim中的word2vec API，官方API介绍如下：

```python
class Word2Vec(utils.SaveLoad):
    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):
```

在`gensim`中，`word2vec`相关的`API`都在包`gensim.models.word2vec`中。和算法有关的参数都在类`gensim.models.word2vec.Word2Vec`中。算法需要注意的参数有：

- **sentences**: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出(通过`word2vec`提供的`LineSentence`类来读文件，`word2vec.LineSentence(filename)`)。
- **vector_size**: 词向量的维度，默认值是`100`。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于`100M`的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。
- **window**：即词向量上下文最大距离，`window`越大，则和某一词较远的词也会产生上下文关系。默认值为`5`。在实际使用中，可以根据实际的需求来动态调整这个`window`的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在`[5,10]`之间。
- **sg**: 即我们的`word2vec`两个模型的选择了。如果是`0`， 则是`CBOW`模型，是`1`，则是`Skip-Gram`模型，默认是`0`，即`CBOW`模型。
- **hs**: 即我们的`word2vec`两个解法的选择了，如果是`1`， 则是`Hierarchical Softmax`，是`0`的话并且负采样个数`negative`大于`0`， 则是`Negative Sampling`。默认是`0`即`Negative Sampling`。
- **negative**:即使用`Negative Sampling`时负采样的个数，默认是`5`。推荐在`[3,10]`之间。
- **cbow_mean**: 仅用于`CBOW`在做投影的时候，为`0`，则算法中的xw��为上下文的词向量之和，为`1`则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw��,默认值也是`1`,不推荐修改默认值。
- **min_count**:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是`5`。如果是小语料，可以调低这个值。
- **iter**: 随机梯度下降法中迭代的最大次数，默认是`5`。对于大语料，可以增大这个值。
- **alpha**: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为`η`，默认是`0.025`。
- **min_alpha**: 由于算法支持在迭代的过程中逐渐减小步长，`min_alpha`给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。



在学习如何训练FastText和Word2Vec词向量模型之前，需要准备一个大规模的文本语料库。语料库应包含足够的文本样本，以便模型能够学习到丰富的语义信息。一般来说，语料库可以是大量的文本文档、新闻文章、维基百科数据等。

下面是训练FastText和Word2Vec词向量模型的实践步骤：

1. 准备大规模文本语料库：收集或准备一个适当的大规模文本语料库。可以使用自己的文本数据集，也可以使用公开可用的语料库。

2. 导入相关库和模块：使用Python导入所需的库和模块。对于FastText，你可以使用FastText库；对于Word2Vec，你可以使用gensim库。

   ```python
   # 导入FastText库
   from gensim.models.fasttext import FastText

   # 导入Word2Vec库
   from gensim.models import Word2Vec
   ```

3. 设置模型参数并构建词向量模型：为模型选择适当的参数，例如词向量维度、窗口大小、训练迭代次数等。然后，使用FastText或Word2Vec类创建词向量模型。

   ```python
   # 创建FastText词向量模型
   fasttext_model = FastText(size=100, window=5, min_count=5)

   # 创建Word2Vec词向量模型
   word2vec_model = Word2Vec(size=100, window=5, min_count=5)
   ```

   在上面的示例中，`size` 参数表示词向量的维度，`window` 参数表示词语的上下文窗口大小，`min_count` 参数表示词语在语料库中的最低出现次数。

4. 构建词汇表：使用Word2Vec类的 `build_vocab()` 方法，根据语料库构建词汇表。

   ```python
   # 对FastText模型构建词汇表
   fasttext_model.build_vocab(corpus)

   # 对Word2Vec模型构建词汇表
   word2vec_model.build_vocab(corpus)
   ```

   这里的 `corpus` 是语料库的输入数据，可以是一个迭代器或一个可迭代的文本文件。

5. 训练词向量模型：使用Word2Vec类的 `train()` 方法，对词向量模型进行训练。

   ```python
   # 训练FastText词向量模型
   fasttext_model.train(corpus, total_examples=fasttext_model.corpus_count, epochs=10)
   
   # 训练Word2Vec词向量模型
   word2vec_model.train(corpus, total_examples=word2vec_model.corpus_count, epochs=10)
   ```

   在上面的示例中，`corpus` 是训练数据，`total_examples` 表示语料库的总样本数，`epochs` 表示训练迭代次数。

   训练过程会根据语料库中的文本数据来更新词向量模型的权重和参数，使得模型能够学习到词语之间的语义信息。

完成以上步骤后，你将得到训练好的FastText或Word2Vec词向量模型，可以使用它们来获取词语的向量表示，进行词语相似度计算、文本分类等自然语言处理任务。







参考

- https://xiaomindog.github.io/2021/06/22/word2vec-and-fastText/
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [秒懂词向量Word2vec的本质](https://zhuanlan.zhihu.com/p/26306795)

