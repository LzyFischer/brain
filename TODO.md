# 10.16
## Think
1. 我应该怎么安排aiying的这个project
   1. 先尝试FC



# 10.14
## Think
1. 效果目前趋于正常，还存在问题：不平衡，缺失数据，数据量少
   1. 应该是先拿到全部数据还是site11的数据？-》尽量全部数据
   2. 做什么实验：（1）FC的实验，（2）Fusion
      1. 先**最简单的实验**，如果效果还不错-》直接使用**最好的配置**
         1. 不平衡的数据，填补缺失数据
         2. 方法仍然使用最简单的gnn （可以试试mamba）
      2. **fusion的最简单的实验**和全部数据



# 10.11
## Think
1. 如果用了新的数据还是效果不好就可视化



# 10.4
## Think 
1. 使用graph-mamba跑的结果
   1. performance并没有变好
      1. 52 - 58 range
      2. recall的值的波动相对比较大



# 10.2
## Think 
1. 加了这个feature跑了吗
2. 加了relu在linear之后反而结果变得奇怪了，更多波动
   1. linear时0.5的auc
3. 数据有问题，我甚至用[0,1]的值做label都能有一个好效果 -》 是不是模型的问题？
   1. 把edge再稀疏一点，数据normalize不同
   2. 是不是label的问题？-》没有uncertain的label会有什么影响吗？不会
   3. 最有可能是模型 / 数据的问题
      1. 使用不同的model -》 GAT， SAGE
      2. 效果能高一些
4. 改变训练比例或许有用 -> 没用
5. attention的train的结果也是0.5
   


# 10.1
## Think
1. 类别不平衡在site 11也有这问题-》尤其是总共也就只有30个正类
2. 先试试processing，GIN not working
3. 看看是不是代码写错了，先试试label作为feature
   1. 模型没问题-》100 performance
4. 需要做一个correlation的分析吗？
5. 把其余的feature merge进来怎么样呢？（age）





# 9.28
## Finding
1. 使用macro的结果反而比micro的auc还要更差

## TODO
1. 检查模型没有问题，使用最可能正确的数据集版本。
   1. -1 - 1
      1. 这个结果也很差,甚至不如
2. solution for the imb -> use different pos_weights for different labels -》 加了positive weight反而结果更差了
3. 检查清单
   1. 模型
      1. linear model -> no
   2. training 过程
      1. 1e-2， 1e-4
   3. evaluate metric
   4. 数据集
      1. 使用不同的label，平衡数据集

## Problem
1. bad performance
      1. possible reason: 
         1. x_SC很大
         2. loss很大，但是x——FC相对来说比较小 -> 一般来说只对初始有坏处
            1. 是因为模型的原因吗？-》最简单的模型
               1. 使用linear后test auc -》 0.55， training的auc到了100，test的多了一点
            2. edge 太dense了
               1. 0.8作为threshold之后的结果比linear更差
         3. class imbalance
            1. solution: down sample -> training performs much better, test performs slightly better
      2. finding:
         1. recall和precision都很低
2. performance not stable
   1. random 的话很不稳定 (seed固定)
   2. 不random又没有一个positive value



# 9.27
## TODO
1. 实验安排
   1. 不加weight/加weight-》unbalanced
   2. 不同modality的ablation实验



# 9.26
## Problem
1. 为什么不能够100%
2. 为什么aucroc是0.5
3. 随机的结果
   1. 太sparse了？


## TODO
1. one-modality curve
2. fusion modality curve


# 9.25 
## Problem
1. 应不应该去掉555的项目

## TODO
1. 选用什么方法来实现
    1. base的简单方法
        1. 不用graph直接融合
        2. 用graph
    2. baseline
        1. RH-BrainFS -> 实现有难度
        2. 数据预处理？



# 4.29
## Problem
1. 到底是分类疾病还是年龄
### dgl or pyg?
1.  




# Dataset
## abcd
1. 多分类问题。每一个subject对应一个key和多个疾病