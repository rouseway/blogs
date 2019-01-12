
# 用Keras构建神经网络

构建一个神经网络主要围绕以下四个方面

- **层**：多个层组成一个网络
- **输入与输出**：输入网络的数据以及与每一个输入数据对应的目标值
- **损失函数**：用于学习的反馈信号，在训练过程中不断地将其最小化，可以用它来衡量当前任务是否已经成功完成
- **优化器**：决定学习过程如何进行，即如何基于损失函数对网络进行更新

在Keras中神经网络模型是靠**层**的堆叠构建而成的，有许多不同形式的层如你经常听到的**全连接层**（Keras里叫做**密集连接层Dense**）、**卷积层**、**循环网络层**等等。每一个层只接受特定形状的输入张量，并返回特定形状的输出张量。我们通过向一个**Sequential**模型对象中不断添加相应的层就可以构建网络，Keras很友好的一点就是向模型中添加的层会自动匹配输入层的形状，例如下面这段代码：


```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))  #只有第一层需要定义输入的形状
model.add(layers.Dense(32), activation='sigmoid')  #其余层只需要定义输出的形状
```
关于每一层所使用的激活函数，可以按照如下的建议去设计：


对于损失函数的选择，我们有如下的经验指导原则：

+ **二分类问题**：使用二元交叉熵（binary crossentropy）损失函数
+ **多分类问题**：使用分类交叉熵（category crossentropy）损失函数
+ **回归问题**：使用均方差（mean-squared error）损失函数
+ **序列学习问题**：联结主义时序分类（CTC, connectionist temporal classification）损失函数

当然，除过这些常见任务之外的特定问题可能需要我们自己定义损失函数。对于优化器的选择，无论你的问题是什么，**RMSprop**优化器通常都是足够好的选择。在Keras中，你需要把选择好的损失函数与优化器放入你所构建的网络模型中进行编译，如下面这段代码：


```python
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', 
										metrics=['accuracy'])
```

最后，学习的过程就是通过fit()方法将输入数据的Numpy数组（以及对应的目标值）传入模型：


```python
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
```

以上，就是使用Keras构建神经网络的基本思路，这只是我们用Keras进行深度学习的基本脉络，实践中的方法论我们将在下一节讨论，它们包括如何处理数据、如何调参、如何防止过拟合，以及如何评价模型的好坏。
