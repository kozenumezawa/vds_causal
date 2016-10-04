# coding: utf-8

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import random

def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)

def bias_variable(shape, variable_name):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial, name=variable_name)

def main():
    rawdata = numpy.load('ocean.normalized.npy') # rawdata.shape = (10100, 2, 212)
    s = numpy.zeros((rawdata.shape[0], rawdata.shape[2]), dtype=numpy.float32)
    t = numpy.zeros((rawdata.shape[0], rawdata.shape[2]), dtype=numpy.float32)

    for i in range(rawdata.shape[0]):
        s[i, :rawdata.shape[2]] = rawdata[i, 0]          #   S(salinity)
        t[i, :rawdata.shape[2]] = rawdata[i, 1]          #   T(water temperature)

    # print(st.shape) = (10100, 424)
    PIXELS = s.shape[1]  # = 212
    H = 50
    BATCH_SIZE = 1
    DROP_OUT_RATE = 0.5

    x = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='y_')
    W1 = weight_variable((PIXELS, H), 'W')
    b1 = bias_variable([H], 'b1')

    h = tf.nn.softsign(tf.matmul(x, W1) + b1)
    # h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    # h = tf.matmul(x, W1) + b1
    keep_prob = tf.placeholder("float", name='keep_prob')
    h_drop = tf.nn.dropout(h, keep_prob)

    W2 = tf.transpose(W1)  # 転置
    b2 = bias_variable([PIXELS], 'b2')
    y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

    # loss = tf.nn.l2_loss(y - x) / BATCH_SIZE
    loss = tf.reduce_mean(tf.square(y_ - y) * 10000)
    tf.scalar_summary("l2_loss", loss)

    train_step = tf.train.AdamOptimizer().minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph=sess.graph)

    for step in range(3001):
        data_index = random.randint(1,10099)
        inputdata = numpy.array([s[data_index]])
        trainningdata = numpy.array([t[data_index]])
        sess.run(train_step,
                 feed_dict={x: inputdata, y_: trainningdata, keep_prob: (1 - DROP_OUT_RATE)})
        if step % 100 == 0:
            print(step, loss.eval(session=sess, feed_dict={x: inputdata, y_: trainningdata, keep_prob: 1.0}))
            times = [i for i in range(PIXELS)]
            # output = y.eval(session=sess, feed_dict={x: inputdata,, y: trainningdata keep_prob: 1.0})
        if step % 1000 == 0 and step != 0:
            times = [i for i in range(PIXELS)]
            output = y.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0})
            plt.plot(times, output[0], color='r', lw=2)
            plt.plot(times, trainningdata[0], color='g', lw=1)
            plt.show()
    # # write trainning result
    result_W1 = sess.run(W1)
    numpy.save('../npy/result_W1.npy', result_W1)
    result_b1 = sess.run(b1)
    numpy.save('../npy/result_b1.npy', result_b1)
    result_W2 = sess.run(W2)
    numpy.save('../npy/result_W2.npy', result_W2)
    result_b2 = sess.run(b2)
    numpy.save('../npy/result_b2.npy', result_b2)
if __name__ == '__main__':
    main()
