# Tensorflow
### 正则化的目的

减小过拟合。

### 什么时候要使用激活函数

==中间层的节点输出要使用吗？==

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

`tf.matmul(x,w1)` (记为`M`)应该是一个矩阵，而`b1`是一个向量，我理解的`y1`的结果应该是`M`中的每一行和`b`相加，==可是这行代码表示不了这样的意思啊==？

### 滑动平均是在哪个步骤提高了算法的性能

可以这样理解，滑动平均就是去掉数据的毛刺，过滤掉高频成分。降低训练过程中高频噪声的干扰。

### 交叉熵是在哪里提高了性能

利用交叉熵可以得到更好的损失函数。

### 如何自己拿图片训练，而不是使用库里面的数据？



### 训练的过程究竟是怎么样的？

```python
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		# 
		for i in range (STEP):
			# Get the trainning data from MNIST
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			# Feed the correct input and label to train the Weights and Biases
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
			if i % 100 == 0:
				print ("After %d  training steps, loss on training batch is %g." % (step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

```

`STEP=50000`代表有训练`50000`步，`BATCH_SIZE=200`表示每步从`MNIST`数据集中曲200张数据来训练，那么总共训练的图片数量就应该是`50000*200=1000,0000` 张 ，但是`MNIST`数据的总图片数量都达不到这个数量。

==因此这里的理解哪里出错了？==  

答： 你的理解好像没有错，有人是这样来回答的。感觉有道理。

> maoliyuanjiu 说到： 本来就是一遍又一遍反复训练

另外还有一个问题:

``` python 
_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
```

最开始的`_`==是否代表一个占位==，我其实并不关心这个返回值的大小，放在那里只是为了不把返回值搞错乱掉。



### 测试过程中的计算准确度是怎么计算的？

``` python 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

答: `correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))` 返回的是一个长度为10000的布尔数组。类似这样 `[ True  True  True ...  True  True  True]` 。

接下来`tf.cast(correct_prediction, tf.float32)` 将布尔数组转化成浮点数组`[1 1 1 ... 1 1 1]` ，`1`代表正确，`0`代表错误，然后求和，取平均，进而得到准确度。

**为什么**`correct_prediction`**是一维的呢？**

答: 为什么不是呢？



### MNIST数据集

图片是白底黑字

可是白用什么表示？黑用什么表示？ **值为0.0表示白色，值为1.0表示黑色，值之间表示逐渐变暗的灰色阴影。**

==我认为就不应该二值化，因为MNIST数据集是没有二值化的数据==