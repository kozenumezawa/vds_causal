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

H1 = 35
H2 = 25
DROP_OUT_RATE = 0.5
keep_prob = tf.placeholder("float", name='keep_prob')

W12 = weight_variable((TIME_STEP_2, H1), 'W1')
b12 = bias_variable([H1], 'b1')
h12 = tf.nn.softsign(tf.matmul(x, W12) + b12)
h_drop12 = tf.nn.dropout(h12, keep_prob)

W23 = weight_variable((H1, H2), 'W23')
b23 = bias_variable([H2], 'b23')
h23 = tf.nn.softsign(tf.matmul(h_drop12, W23) + b23)
h_drop23 = tf.nn.dropout(h23, keep_prob)

W34 = tf.transpose(W23)  # 転置
b34 = tf.transpose(b12)
h34 = tf.nn.softsign(tf.matmul(h_drop23, W34) + b34)
h_drop34 = tf.nn.dropout(h34, keep_prob)

W45 = tf.transpose(W12)  # 転置
b45 = bias_variable([TIME_STEP_2], 'b45')
y = tf.nn.relu(tf.matmul(h_drop34, W45) + b45)

# loss = tf.nn.l2_loss(y - x) / BATCH_SIZE
loss = tf.reduce_mean(tf.square(y - x) * 100)

tf.scalar_summary("l2_loss", loss)

train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph_def=sess.graph_def)

DATA_NUM = 3001
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
    if step % 1000 == 0 and step != 0:
        output = y.eval(session=sess, feed_dict={x: [inputdata[0]], keep_prob: 1.0})
        plt.plot(times, inputdata[0], color='r', lw=2)
        plt.plot(times, output[0], color='g', lw=1)
        plt.show()
