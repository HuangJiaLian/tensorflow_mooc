# coding:utf8 
# 0.导入模块，生成模拟数据集
import tensorflow as tf 
import numpy as np 
BATCH_SIZE = 8
SEED = 23455

# 基于SEED产生随机数
rdm = np.random.RandomState(SEED)
# 随机数返回32行2列的矩阵 表示32组 体积和重量　作为输入数据集
X = rdm.rand(32,2)
# 自定义一个标签?
Y_ = [[x1 + x2 + (rdm.rand()/10.0 - 0.05)] for (x1, x2) in X]


# 1. 定义神经网络的输入，参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2)) # None:表示不知道有多少组特征数据　
										      # 2:表示每组特征里面包含两项
y_ = tf.placeholder(tf.float32, shape=(None, 1)) # 标签 1: 只有一项表示合格还是不合格

w1 = tf.Variable(tf.random_normal([2,1], stddev = 1, seed = 1))
y = tf.matmul(x, w1)

# 2. 定义损失函数及反向传播方法
loss_mse = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)


# 3. 生成会话，训练STEPS轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	# 训练模型
	STEPS = 20000
	for i in range(STEPS):
		start = (i * BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict = {x: X[start:end], y_: Y_[start:end]})
		if i % 500 == 0:
			print "After %d training steps, w1 is:"  %(i)
			print sess.run(w1),"\n"
	# 输出训练后的参数取值
	print "Final w1 is: \n", sess.run(w1)