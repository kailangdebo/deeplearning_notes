
# deeplearning_notes

-----------------

##[第一周] introduction to deep learning

### 1.1 欢迎来到深度学习工程师微专业

### 1.2 什么是神经网络？

### 1.3 用神经网络进行监督学习

- `三种监督学习`
- standard NN、Convolutional NN、Recurrent NN
- `structured data & unstructured data`
- 有否清晰的定义（）。音频、图像、文字就是非结构化数据。

### 1.4 为什么深度学习会兴起？

![](images/0.png)

- 数据规模影响神经网络的训练。训练集不大的时候，机器学习其它算法都差不多。训练集大的时候，NN有优势

![](images/1.png)

- `data`
- `computation`
- `algorithms`

计算能力越快,可迭代的速度更快。算法也能提高速度

### 1.5 关于这门课

---------------------

## [第二周]神经网络基础

### 2.1 二分分类（binary classification）

![](images/3.png)

- 每张图片分红绿蓝三个矩阵。
- `64 * 64`像素的图片，每张图片的特征向量`x.size`=64 * 64 * 3=12288，也就是 `nx`

![](images/2.png)

- `（x,y）`表示单独一个样本，x是`n_x`维的特征向量，y值为{0,1}
- 训练集由`m`个训练样本组成。
- `M_train`:训练样本个数
- `M_test`:测试样本个数
- 使用大写`X`表示整个数据集，size=（nx，m）
- y.size=(1,m)

### 2.2 logistic回归

![](images/4.png)

- `sigmoid` 	
- 已知x。求y_hat=P(y=1|x)
- output--> y_hat=sigmoid(w_T*x+b)
- sigmoid(z) =1/(1+e^(-z))

### 2.3 logistic 回归损失函数

![](images/5.png)

- **loss function**损失函数 `L(y_hat,y)`:衡量单个训练样本上的表现。
- **cost function**成本函数 `J(w,b)`：所有样本的损失函数

### 2.4 梯度下降法

![](images/6.png)

![](images/7.png)

### 2.5 导数

### 2.6 更多导数的例子

### 2.7 计算图

### 2.8 计算图的导数计算

![](images/8.png)


### 2.9 logistic 回归中的梯度下降法

![](images/9.png)

### 2.10 m 个样本的梯度下降

![](images/10.png)

### 2.11 向量化

- 为了消除过多for loop。深度学习需要使用向量化

### 2.12 向量化的更多例子

### 2.13 向量化 logistic 回归

### 2.14 向量化 logistic 回归的梯度输出

![](images/11.png)

### 2.15 Python 中的广播

### 2.16 关于 python / numpy 向量的说明

### 2.17 Jupyter / Ipython 笔记本的快速指南

### 2.18 （选修）logistic 损失函数的解释

## [第三周]浅层神经网络

### 3.1 神经网络概览

![](images/12.png)

### 3.2 神经网络表示

![](images/13.png)

- 上图是2层神经网络，因为不算输入层

### 3.3 计算神经网络的输出

![](images/14.png)

![](images/16.png)

![](images/15.png)

### 3.4 多个例子中的向量化

![](images/17.png)

![](images/18.png)

### 3.5 向量化实现的解释

![](images/19.png)

![](images/20.png)

### 3.6 激活函数

![](images/21.png)

![](images/22.png)

- sigmoid tanh reLu（修正线性单元，默认单元了）
- tanh函数比sigmoid函数更优越
- 除非在判断对错等情况
- 选择激活函数的经验法则：
- 如果你的输出值是0或1，如果你在做二元分类，选sigmoid，然后其他所有单元都用ReLU 

### 3.7 为什么需要非线性激活函数？

### 3.8 激活函数的导数

![](images/23.png)

![](images/24.png)

![](images/25.png)

### 3.9 神经网络的梯度下降法

![](images/26.png)

### 3.10 （选修）直观理解反向传播

![](images/27.png)

![](images/28.png)

### 3.11 随机初始化

![](images/29.png)

- 如果设置所有权重初始化为0，则不同的隐藏单元计算结果一直都是一样的。多个隐藏单元就没意义了。
- 解决办法就是随机初始化所有隐藏单元。
- 我们通常喜欢将权重矩阵初始化成很小的数。所以随机数后*0.01。因为如果数值大，落在梯度很少的地方，训练的速度就会慢。

## [第四周]深层神经网络

### 4.1 深层神经网络

![](images/30.png)

### 4.2 深层网络中的前向传播

![](images/31.png)

### 4.3 核对矩阵的维数

![](images/32.png)

![](images/33.png)

### 4.4 为什么使用深层表示

![](images/34.png)

### 4.5 搭建深层神经网络块

![](images/35.png)

![](images/36.png)

### 4.6 前向和反向传播

### 4.7 参数 VS 超参数

![](images/37.png)

### 4.8 这和大脑有什么关系？

