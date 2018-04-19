import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200

# regulate learning rate along with training step
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99

# avoid over fitting 
REGULARIZER = 0.0001
STEP = 50000

# 
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME='mnist_model' 


def backward(mnist):
	# Input nodes
	x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
	# The correct standard output
	y_ = tf.placeholder(tf.float32,[None, mnist_forward.OUTPUT_NODE])
	# Output nodes
	y = mnist_forward.forward(x, REGULARIZER)
	
	global_step = tf.Variable(0,trainable = False)

	# Define the loss function
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))

	# Define the dynamic learning rate
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase = True
		)
	# Use gradient descent optimizer to train the weights and biases 
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

	# Use moving avarage to avoid the noises 
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name = 'train')

	# Define a saver to sava our trained results.
	saver = tf.train.Saver()

	# The process of tensorflow:
	# Define the functions and variables first, 
	# then control how to use those functions and 
	# variables in tf.Session
	with tf.Session() as sess:
		# Must have global_variables_initializer if 
		# you have defined the tf.Variable before 
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		# Restore the trainnig step 
		# if it was been interrupted (Very Cool)
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		for i in range (STEP):
			# Get the trainning data from MNIST
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			# Feed the correct input and label to train the Weights and Biases
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
			if i % 1000 == 0:
				print ("After %d  training steps, loss on training batch is %g." % (step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)


def main():
	mnist = input_data.read_data_sets("./data/", one_hot = True)
	backward(mnist)

if __name__ == '__main__':
	main()
