# coding: utf-8

import matplotlib.pyplot as plt
import numpy
import csv
import tensorflow as tf

def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)


def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)

# load test data
testrawdata = numpy.load('../npy/input_test_data.npy')
testdata = numpy.zeros((testrawdata.shape[0], testrawdata.shape[2] * testrawdata.shape[1]), dtype=numpy.float32)
for i in range(testrawdata.shape[0]):
    testdata[i,:testrawdata.shape[2]] = testrawdata[i, 0]       #   X
    testdata[i, testrawdata.shape[2]:] = testrawdata[i, 1]      #   Y

# load trainning data
rawdata = numpy.load('../npy/input_logistic_data.npy')
# (10000,2,100)→(10000,200)  (to learn causality)
inputdata = numpy.zeros((rawdata.shape[0], rawdata.shape[2] * rawdata.shape[1]), dtype=numpy.float32)
for i in range(rawdata.shape[0]):
    inputdata[i,:rawdata.shape[2]] = rawdata[i, 0]       #   X
    inputdata[i, rawdata.shape[2]:] = rawdata[i, 1]      #   Y

BATCH_SIZE = 1
TIME_STEP_2 = inputdata.shape[1]

x = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEP_2], name='x')

MIDDLE_UNIT = 30
W1 = weight_variable((TIME_STEP_2, MIDDLE_UNIT), 'W1')
b1 = bias_variable([MIDDLE_UNIT], 'b1')

DROP_OUT_RATE = 0.5

h = tf.nn.softsign(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W) + b1
# h = tf.nn.sigmoid(tf.matmul(x, W) + b1)

keep_prob = tf.placeholder("float", name='keep_prob')
h_drop = tf.nn.dropout(h, keep_prob)

W2 = tf.transpose(W1)  # 転置
#W2 = weight_variable((MIDDLE_UNIT, TIME_STEP_2), 'W2')
b2 = bias_variable([TIME_STEP_2], 'b2')
y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

# loss = tf.nn.l2_loss(y - x) / BATCH_SIZE
loss = tf.reduce_mean(tf.square(y - x) * 100)

tf.scalar_summary("l2_loss", loss)

train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph=sess.graph)

DATA_NUM = 2001
times = [i for i in range(TIME_STEP_2)]
# trainning loop
for step in range(DATA_NUM):
    sess.run(train_step,
             feed_dict={x: [inputdata[step]], keep_prob: (1 - DROP_OUT_RATE)})
    summary_op = tf.merge_all_summaries()
    summary_str = sess.run(summary_op, feed_dict={x: [inputdata[step]], keep_prob: 1.0})
    summary_writer.add_summary(summary_str, step)
    if step % 100 == 0:
        train_accuracy = loss.eval(session=sess, feed_dict={x: [inputdata[step]], keep_prob: 1.0})
        print("step %d:%g" % (step, train_accuracy))
    # if step % 1000 == 0 and step != 0:
    #     output = y.eval(session=sess, feed_dict={x: [inputdata[0]], keep_prob: 1.0})
    #     plt.plot(times, inputdata[0], color='r', lw=2)
    #     plt.plot(times, output[0], color='g', lw=1)
    #     plt.show()
# write trainning result
result_W1 = sess.run(W1)
numpy.save('../npy/result_W1.npy', result_W1)
result_b1 = sess.run(b1)
numpy.save('../npy/result_b1.npy', result_b1)
result_W2 = sess.run(W2)
numpy.save('../npy/result_W2.npy', result_W2)
result_b2 = sess.run(b2)
numpy.save('../npy/result_b2.npy', result_b2)
