# coding:utf8
import tensorflow as tf 

# 1x2的张量
x = tf.constant([[1.0, 2.0]])
# 2x1的张量
w = tf.constant([[3.0], [4.0]])

y = tf.matmul(x, w)
print y