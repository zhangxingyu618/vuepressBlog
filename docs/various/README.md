## 01 爬虫

利用 xpath 进行解构爬取数据，获取下一页的 url，不断循环爬取所有新闻文章。

## 02 jieba 分词、词云

#### 先读取 csv 文件中的新闻数据的标题和正文内容到 txt 文件中。

#### 然后进行 jieba 分词处理：

- 精确模式，试图将句子最精确地切开，适合文本分析；
- 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
- 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。

```python
import jieba

jieba.cut(data, cut_all=False) # 精准模式
jieba.cut(data, cut_all=True) # 全模式
jieba.cut_for_search(data)	 # 搜索引擎模式
```

#### 分词之后进行简单的词频统计，此时过滤掉了字符串长度为 1 的字符。

#### 利用 pyecharts 进行绘制词云图，被保存 html 页面文件

## 03 TF-IDF

#### TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率).

##### TFIDF 的主要思想是：如果某个词或短语在一篇文章中出现的频率 TF 高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类

**词频 (term frequency, TF)** 指的是某一个给定的词语在该文件中出现的次数。

**逆向文件频率 (inverse document frequency, IDF)** 是一个词语普遍重要性的度量。某一特定词语的 IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。

##### IDF = log（语料库中文档总数 / 包含该词的文档数 +1 ）

##### TFIDF 实际上是：TF \* IDF

##### 基于 TF-IDF（term frequency–inverse document frequency） 算法的关键词抽取

```python
import jieba.analyse

jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
```

- **sentence** ：为待提取的文本
- **topK**： 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
- **withWeight** ： 为是否一并返回关键词权重值，默认值为 False
- **allowPOS** ： 仅包括指定词性的词，默认值为空，即不筛选

## 04 K-means 聚类

k：要得到的簇的个数

质心：均值，向量各维取平均值即可，不断更新的

距离的度量：常用欧式距离和余弦的相似度

```python
	# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来 元素a[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()
```

我们可以通过用 TD－IDF 衡量每个单词在文件中的重要程度。如果多个文件，它们的文件中的各个单词的重要程度相似，我就可以说这些文件是相似的。如何评价这些文件的相似度呢？一种很自然的想法是用两者的欧几里得距离来作为相异度，欧几里得距离的定义如下：

![image-20210103154152791](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210103154152791.png)

其意义就是两个元素在欧氏空间中的集合距离，因为其直观易懂且可解释性强，被广泛用于标识两个标量元素的相异度。我们可以将 X，Y 分别理解为两篇文本文件，xi,y 是每个文件单词的 TD－IDF 值。这样就可以算出两文件的相似度了。这样我们可以将文件聚类的问题转化为一般性的聚类过程，样本空间中的两点的距离可以欧式距离描述。除欧氏距离外，常用作度量标量相异度的还有曼哈顿距离和闵可夫斯基距离，两者定义如下：

![image-20210103154217134](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210103154217134.png)

整个文本聚类过程可以先后分为两步：1、计算文本集合各个文档中 TD－IDF 值，2，根据计算的结果，对文件集合用 k-means 聚类方法进行迭代聚类。

有关 k-means 的详细介绍 https://blog.csdn.net/freesum/article/details/7376006

## 05 层次聚类

层次聚类的合并算法通过计算两类数据点间的相似性，对所有数据点中最为相似的两个数据点进行组合，并反复迭代这一过程。简单的说层次聚类的合并算法是通过计算每一个类别的数据点与所有数据点之间的距离来确定它们之间的相似性，距离越小，相似度越高。并将距离最近的两个数据点或类别进行组合，生成聚类树。

#### 基本步骤

1.计算每两个观测之间的距离

2.将最近的两个观测聚为一类，将其看作一个整体计算与其它观测(类)之间的距离

3.一直重复上述过程，直至所有的观测被聚为一类

#### 例子

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
print(X)
Z = linkage(X, 'ward')
print(Z)
f = fcluster(Z, 4, 'distance')
print(f)
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()
```

![img](https://img-blog.csdnimg.cn/20190301181326559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lpYm80OTIzODc=,size_16,color_FFFFFF,t_70)

## 06 lda 主题模型

LDA（Latent Dirichlet Allocation）是一种文档主题生成模型，也称为一个三层贝叶斯概率模型，包含词、主题和文档三层结构。所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布。

```python
import pyLDAvis
# pyLDA需要先导入模型，支持的模型的来源有三种：
	# sklearn的lda模型 （我们用的这种）
	# gensim的lda模型
	# graphlab的lda模型
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
```

文档主题生成模型（Latent Dirichlet Allocation，简称 LDA）又称为盘子表示法（Plate Notation），下图是模型的标示图，其中双圆圈表示可测变量，单圆圈表示潜在变量，箭头表示两个变量之间的依赖关系，矩形框表示重复抽样，对应的重复次数在矩形框的右下角显示。LDA 模型的具体实现步骤如下：

> 从每篇网页 D 对应的多项分布 θ 中抽取每个单词对应的一个主题 z。

> 从主题 z 对应的多项分布 φ 中抽取一个单词 w。

> 重复步骤 1 和 2，共计 Nd 次，直至遍历网页中每一个单词。

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210103161446056.png" alt="image-20210103161446056" style="zoom: 50%;" />

##### 有关 LDA 只提模型原理的介绍 https://zhuanlan.zhihu.com/p/31470216

##### 在生成的网页中的相关知识

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210103162718104.png" alt="image-20210103162718104" style="zoom:80%;" />

浅蓝色的表示这个词在整个文档中出现的频率（权重），深红色的表示这个词在这个主题中所占的权重。

如果`λ`接近 1，那么在该主题下更频繁出现的词，跟主题更相关；
如果`λ`越接近 0，那么该主题下更特殊、更独有的词，跟主题更相关（有点 TF-IDF 的意思了）。

所以我们可以通过调节`λ`的大小来改变词语跟主题的相关性，探索更加合理的主题意义。

## 07 结论

在大数据时代下，当运用传统的数学方法遇到困难时，熟练地应用数据挖掘技术显得格外重要。文本数据挖掘并不是一件容易的事情，尤其是在分析方法方面，还有很多需要研究的专题。随着计算机计算能力的发展和业务复杂性的提高，数据的类型会越来越多、越来越复杂，数据挖掘将发挥出越来越大的作用。

经过大家的不懈努力，对疫情新闻的相关分析也已经完成。

通过本次对该项目的研究，我们利用文本挖掘将大量繁琐复杂的新闻通过词云的方式进行展示，以及在数据预处理上对其采集的数据进行了数据清洗，得到了更标准、高质量的数据来提升分析的结果。

总的体会可以用一句话来表达，纸上得来终觉浅，绝知此事要躬行!通过对中国社会组织公共服务平台的爬虫，我们也是了解到了许多在疫情期间让人感概落泪的故事以及政府在全力保护人民安全和健康所做出的一系列决策。

从 tf-idf 权重计算和词云的展示中可以看到可以看到“疫情”、“组织”、“捐赠”、“社会”、“协会”、“肺炎”、“复工”等都是社会和政府共同关注的主题；

在层次聚类中可以看出不同领域的关键词有着不同的聚类最终也都汇聚成一块；

我们还发现在 LDA 主题模型中输出结果分成了两类，在某种意义上可理解为一类是疫情，一类是民生，也表明了政府在防疫、控制疫情蔓延的同时十分关心人民群众的生活，在中国政府的眼里，人民的健康和安全永远摆在第一位！
