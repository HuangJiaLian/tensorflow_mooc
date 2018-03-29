# coding:utf8
# 这就是一个计算图(Graph):
# 搭建神经网络的计算过程，只搭建，不运算

import tensorflow as tf

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

result = a + b

# 计算图只描述运算过程，不计算运算结果
print result


