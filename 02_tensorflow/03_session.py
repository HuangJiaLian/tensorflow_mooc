# coding:utf8
# 会话(Session): 执行计算图中的节点运算
import tensorflow as tf 

# 1x2的张量
x = tf.constant([[1.0, 2.0]])
# 2x1的张量
w = tf.constant([[3.0], [4.0]])

y = tf.matmul(x, w)
print y

with tf.Session() as sess:
	print sess.run(y)
