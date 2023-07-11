# Coggle 30 Days of ML（23年7月）任务五：XGBoost训练与预测



#### 任务五：使用TFIDF特征和XGBoost完成训练和预测

- 说明：在这个任务中，你需要使用TFIDF特征和XGBoost算法完成训练和预测，进一步提升文本分类的性能。
- 实践步骤：
  1. 准备TFIDF特征矩阵和相应的标签。
  2. 划分训练集和测试集。
  3. 使用Sklearn中的XGBoost算法进行训练，并使用训练好的模型对测试集进行预测。
  4. 评估模型的性能，如准确率、精确率、召回率等指标。

> 对于这一部分来说，我们只需要换一下模型即可，从线性模型换成一个比较强大的XGBoost模型即可完成，所以主要在评估结果中
>
> 

## TFIDF提取特征

首先使用任务三中的方法先提取特征

```python
tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w\w+\b', max_features=4000, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train_data['content'])
test_tfidf = tfidf.fit_transform(test_data['content'])
```

这样我们就一句得到了TFIDF的特征矩阵，接下来我们就可以进行下一步的训练和测试了



## 训练XGBoost

这里修改为训练XGBoost模型

```python
model = xgb.XGBClassifier()
model.fit(train_tfidf, train_data['label'])
```

训练完以后，我们就得到了一个不错的XGB模型，接下来我们可以进行评估模型的性能



## 评估模型

首先我们可以计算一下准确率，从结果上来看，准确率很圆满为100%

```python
predictions = model.predict(train_tfidf)
accuracy = accuracy_score(train_data['label'], predictions)
print("Accuracy:", accuracy)
```

```bash
Accuracy: 1.0
```

我们还计算了精确率和召回率的指标，均为100%

```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(train_data['label'], predictions)
recall = recall_score(train_data['label'], predictions)
print("Precision:", precision)
print("Recall:", recall)
```

```bash
Precision: 1.0
Recall: 1.0
```



## 模型预测及提交

最后利用模型对测试集进行预测，得到结果文件

```python
submit = pd.read_csv('ChatGPT/sample_submit.csv')
submit = submit.sort_values(by='name')

submit['label'] = model.predict(test_tfidf).astype(int)

submit.to_csv('ChatGPT/xgb.csv', index=None)
```

经过提交以后，最后的分数为0.8848，从结果上来看，还是存在一些过拟合的，还是需要对其进行一些调参以得到更好的结果，并且可能还是需要一个验证集来检测结果是否过拟合，减小过拟合应该可以得到不错的分数

![在这里插入图片描述](https://img-blog.csdnimg.cn/6baa601746374f0580114ddbe60426e5.png)



## 改进与提高

后续我进行学习的时候，再进行了改进，得到了更好的结果，我加入了cross_val_predict()函数进行交叉验证，使用交叉验证的目的是为了更准确地评估模型的性能和泛化能力。交叉验证是一种评估机器学习模型的统计方法，通过将数据集划分为多个子集，在不同的训练和验证集上多次训练和评估模型，以得到更可靠的性能指标。

```python
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
train_data = pd.read_csv('ChatGPT/train.csv')
test_data = pd.read_csv('ChatGPT/test.csv')

# train_data['content'] = train_data['content'].apply(get_num)
# test_data['content'] = test_data['content'].apply(get_num)

train_data['content'] = train_data['content'].apply(lambda x: x[1:-1])
test_data['content'] = test_data['content'].apply(lambda x: x[1:-1])

tfidf = TfidfVectorizer(token_pattern=r'\w{1}',ngram_range=(1,3), max_features=10000)
tfidf.fit(train_data['content'].apply(lambda x: ' '.join(x)))
train_tfidf_feat = tfidf.transform(train_data['content'].apply(lambda x: ' '.join(x)))
test_tfidf_feat = tfidf.transform(test_data['content'].apply(lambda x: ' '.join(x)))

val_pred = cross_val_predict(
    XGBClassifier(n_estimators=50),
    train_tfidf_feat,
    train_data['label']
)
print(classification_report(train_data['label'], val_pred, digits=3))

m = XGBClassifier(n_estimators=50)
m.fit(train_tfidf_feat, train_data['label'])
test_pred = m.predict(test_tfidf_feat)
test_data['label'] = test_pred
test_data[['name', 'label']].to_csv('ChatGPT/xgb.csv', index=None)
```

```python
              precision    recall  f1-score   support

           0      0.974     0.995     0.984     11836
           1      0.966     0.852     0.905      2164

    accuracy                          0.973     14000
   macro avg      0.970     0.923     0.945     14000
weighted avg      0.972     0.973     0.972     14000
```

最后得到了更好的成绩，97.36，后续继续努力，冲上99+！！！

![在这里插入图片描述](https://img-blog.csdnimg.cn/ff5ef36f5c604f65bc3ab2e3a53d8aea.png)

最后我有对模型进行调整，最后又有提升，还差一点点到99+了，继续努力，我觉得xgboost的潜力还是很大的，单单xgboost肯定就有99+的潜力，还可以继续冲一下

![在这里插入图片描述](https://img-blog.csdnimg.cn/0230f811412e4bc896378770e2837517.png)