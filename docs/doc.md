## 完成后请注明已完成
### 数据统计（完成后将变量的值写入文档
* Bert(多语言模型）词向量维数$d_w=768$
* 数据条数 $N=148075238$
* 语言总数 $N_L=66$
* Hashtag总数 $N_H=2554184$
* 最大关注者数（engaged和engaging不做区分一起统计，下同) $M_{FER}=112787050$
* 最大关注数 $M_{FNG}=4447036$
* Hashtag分布 **TODO**
* ~~关注者数分布~~
* ~~关注数分布~~
### 数据处理格式
每一条数据是一个*list*,每一维分别为：  
1. $d_w$维向量，实数值，为text_token id对应词向量的平均值
2. ~~~$N_H$维向量，multi-hot,表示有哪些tag(待定）~~~
3. $3$维向量，multi-hot,依次对应(Photo,Video,Gif)
4. $4$维向量，one-hot,依次对应(Retweet,Quote,Reply,Toplevel)
5. $N_L$维向量，one-hot,表示对应语言
6. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FER}\ +1)}$, 归一化后的发推者关注者数
7. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FNG}\ +1)}$, 归一化后的发推者关注数
8. $2$维向量，one-hot,依次对应（False,True),表示发推者是否经过验证
9. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FER}\ +1)}$, 归一化后的互动者关注者数
10. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FNG}\ +1)}$, 归一化后的互动者关注数
11. $2$维向量，one-hot,依次对应（False,True),表示互动者是否经过验证
12. $2$维向量，one-hot,依次对应（False,True),表示发推者是否关注互动者
13. $1$维向量，True or False,表示是否产生reply
14. $1$维向量，True or False,表示是否产生retweet
15. $1$维向量，True or False,表示是否产生retweet_with_comment
16. $1$维向量，True or False,表示是否产生like

一个能将一条原始数据转成如上格式的函数  
一个能将一个原始数据文件转化成如上格式文件的函数

### 小数据
* 在服务器lzh/Challenge/data目录下有两组小数据
* toy_training.tsv/toy_val.tsv 分别有1000条数据用于dubug
* reduced_training.tsv/reduced_val.tsv 分别有$10^6$和$10^5$条数据用于release(看效果)

### 文件和目录解释
* data目录用于存放twitter数据
  * tsv文件为原数据格式
  * 处理后的数据以npy格式存放
* bert-base-multilingual-cased用于存放bert
* config.py用于parser以及存放所有环境参数和超参数
* data_process.py用于处理数据
* data_loader.py用于生成dataset,向模型提供数据
* train.py用于训练
* test.py用于测试
* 每个模型单独一个文件用模型名字命名（如FM.py)
