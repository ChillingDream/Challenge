## 完成后请注明已完成
### 数据统计（完成后将变量的值写入文档）TODO
* Bert(多语言模型）词向量维数$d_w$
* 数据条数 $N$
* 语言总数 $N_L$
* Hashtag总数 $N_H$
* 最大关注者数（engaged和engaging不做区分一起统计，下同) $M_{FER}$
* 最大关注数 $M_{FNG}$
* 关注者数分布
* 关注数分布
### 数据处理格式
每一条数据是一个*list*,每一维分别为：  
1. $d_w$维向量，实数值，为text_token id对应词向量的平均值
2. $N_H$维向量，multi-hot,表示有哪些tag(待定）
3. $3$维向量，multi-hot,依次对应(photo,video,gif)
4. $4$维向量，one-hot,依次对应(retweet, quote, reply, toplevel)
5. $N_L$维向量，one-hot
6. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FER}\ +1)}$, 归一化后的发推者关注者数
7. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FNG}\ +1)}$, 归一化后的发推者关注数
8. $2$维向量，one-hot,依次对应（True,False),表示发推者是否经过验证
9. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FER}\ +1)}$, 归一化后的互动者关注者数
10. $1$维向量，实数，$normalize(c)=\frac{log(c+1)}{log(M_{FNG}\ +1)}$, 归一化后的互动者关注数
11. $2$维向量，one-hot,依次对应（True,False),表示互动者是否经过验证
12. $2$维向量，one-hot,依次对应（True,False),表示发推者是否关注互动者
13. $1$维向量，True or False,表示是否产生reply
14. $1$维向量，True or False,表示是否产生retweet
15. $1$维向量，True or False,表示是否产生retweet_with_comment
16. $1$维向量，True or False,表示是否产生like
