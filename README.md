# tensorflow_mooc
### 正则化的目的

减小过拟合。

### 什么时候要使用激活函数

中间层的节点输出要使用吗？

``` python
# build the neural network
def forward(x, regularizer):
	w1 = get_weight([INPUT_NODE,LAYER1_NODE], regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x,w1) + b1) # Why use relu at this place?
	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1,w2) + b2 
	return y 
```

另外在上述代码中的

```python
y1 = tf.nn.relu(tf.matmul(x,w1) + b1)
```

`tf.matmul(x,w1)` (记为`M`)应该是一个矩阵，而`b1`是一个向量，我理解的`y1`的结果应该是`M`中的每一行和`b`相加，可是这行代码表示不了这样的意思啊？