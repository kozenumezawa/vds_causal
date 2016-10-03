# coding: utf-8

# input pair of data and learn
# this program focuses on
# S and T(Salinity causes Temperature) and VS(Velocity causes Salinity)

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import random

def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)

def bias_variable(shape, variable_name):
    initial = tf.constant(0.3, shape=shape)
    return tf.Variable(initial, name=variable_name)

def main():
    rawdata = numpy.load('ocean.normalized.npy') # rawdata.shape = (10100, 2, 212)
    data = numpy.zeros((rawdata.shape[0] * 2, rawdata.shape[2] * 2),
                       dtype=numpy.float32)
    st = numpy.zeros((rawdata.shape[0], rawdata.shape[2] * 2), dtype=numpy.float32)

    for i in range(rawdata.shape[0]):
        st[i, :rawdata.shape[2]] = rawdata[i, 0]          #   S(salinity)
        st[i, rawdata.shape[2]:] = rawdata[i, 1]          #   T(water temperature)

    PIXELS = data.shape[1]  # = 424
    H1 = 200
    H2 = 50
    BATCH_SIZE = 1
    DROP_OUT_RATE = 0.5

    x = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x')
    # create network (1st)
    W12 = weight_variable((PIXELS, H1), 'W12')
    b12 = bias_variable([H1], 'b12')
    h12 = tf.nn.sigmoid(tf.matmul(x, W12) + b12)

    keep_prob = tf.placeholder("float", name='keep_prob')
    h_drop12 = tf.nn.dropout(h12, keep_prob)

    W45 = tf.transpose(W12)  # 転置
    b45 = bias_variable([PIXELS], 'b45')
    y_1 = tf.nn.relu(tf.matmul(h_drop12, W45) + b45)

    # loss = tf.nn.l2_loss(y - x) / BATCH_SIZE
    loss = tf.reduce_mean(tf.square(y_1 - x) * 10000)

    train_step = tf.train.AdamOptimizer().minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # first learn
    for step in range(101):
        data_index = random.randint(0,10099)
        inputdata = numpy.array([st[data_index]])
        sess.run(train_step,
                 feed_dict={x: inputdata, keep_prob: (1 - DROP_OUT_RATE)})
        if step % 100 == 0:
            print(step, loss.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0}))
            times = [i for i in range(PIXELS)]
            output = y_1.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0})
        if step % 1000 == 0 and step != 0:
            times = [i for i in range(PIXELS)]
            output = y_1.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0})
            plt.plot(times, inputdata[0], color='r', lw=2)
            plt.plot(times, output[0], color='g', lw=1)
            plt.show()
    # save
    learned_W12 = sess.run(W12)
    learned_b12 = sess.run(b12)
    learned_W45 = sess.run(W45)
    learned_b45 = sess.run(b45)

    keep_W12 = tf.placeholder("float", name='W12')
    keep_b12 = tf.placeholder("float", name='b12')
    keep_W45 = tf.placeholder("float", name='W45')
    keep_b45 = tf.placeholder("float", name='b45')

    # create network (2nd)
    h12_2 = tf.nn.softsign(tf.matmul(x, keep_W12) + keep_b12)

    W23 = weight_variable((H1, H2), 'W23')
    b23 = bias_variable([H2], 'b23')
    h23 = tf.nn.softsign(tf.matmul(h12_2, W23) + b23)
    h_drop23 = tf.nn.dropout(h23, keep_prob)

    W34 = tf.transpose(W23)  # 転置
    b34 = bias_variable([H1], 'b12')
    h34 = tf.nn.softsign(tf.matmul(h_drop23, W34) + b34)
    h_drop34 = tf.nn.dropout(h34, keep_prob)

    y_2 = tf.nn.relu(tf.matmul(h_drop34, keep_W45) + keep_b45)
    loss = tf.reduce_mean(tf.square(y_2 - x) * 10000)

    train_step = tf.train.AdamOptimizer().minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    tf.scalar_summary("l2_loss", loss)
    summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph=sess.graph)

    # second learn
    for step in range(101):
        data_index = random.randint(0,10099)
        inputdata = numpy.array([st[data_index]])
        feed_dict = {
            x: inputdata,
            keep_prob: (1 - DROP_OUT_RATE),
            keep_W12: learned_W12,
            keep_b12: learned_b12,
            keep_W45: learned_W45,
            keep_b45: learned_b45
        }
        sess.run(train_step,
                 feed_dict=feed_dict)
        summary_op = tf.merge_all_summaries()
        summary_str = sess.run(summary_op, feed_dict={x: inputdata, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, step)
        if step % 100 == 0:
            feed_dict = {
                x: inputdata,
                keep_prob: 1.0,
                keep_W12: learned_W12,
                keep_b12: learned_b12,
                keep_W45: learned_W45,
                keep_b45: learned_b45
            }
            print(step, loss.eval(session=sess, feed_dict=feed_dict))
            times = [i for i in range(PIXELS)]
            output = y_2.eval(session=sess, feed_dict=feed_dict)
        if step % 1000 == 0 and step != 0:
            feed_dict = {
                x: inputdata,
                keep_prob: 1.0,
                keep_W12: learned_W12,
                keep_b12: learned_b12,
                keep_W45: learned_W45,
                keep_b45: learned_b45
            }
            times = [i for i in range(PIXELS)]
            output = y_2.eval(session=sess, feed_dict=feed_dict)
            plt.plot(times, inputdata[0], color='r', lw=2)
            plt.plot(times, output[0], color='g', lw=1)
            plt.show()

if __name__ == '__main__':
    main()
