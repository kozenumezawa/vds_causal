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
    H1 = 300
    H2 = 100
    H3 = 25
    BATCH_SIZE = 1
    DROP_OUT_RATE = 0.5

    x1 = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x1')
    # create network (1st)
    W12 = weight_variable((PIXELS, H1), 'W12')
    b12 = bias_variable([H1], 'b12')
    # h12 = tf.nn.softsign(tf.matmul(x1, W12) + b12)
    h12 = tf.matmul(x1, W12) + b12

    keep_prob1 = tf.placeholder("float", name='keep_prob1')
    h_drop12 = tf.nn.dropout(h12, keep_prob1)

    W45 = tf.transpose(W12)  # 転置
    b45 = bias_variable([PIXELS], 'b45')
    y_1 = tf.nn.relu(tf.matmul(h_drop12, W45) + b45)

    # loss1 = tf.nn.l2_loss(y - x) / BATCH_SIZE
    loss1 = tf.reduce_mean(tf.square(y_1 - x1) * 10000)

    train_step1 = tf.train.AdamOptimizer().minimize(loss1)
    init = tf.initialize_all_variables()
    sess1 = tf.Session()
    sess1.run(init)

    # first learn
    for step in range(10001):
        data_index = random.randint(0,10099)
        inputdata = numpy.array([st[data_index]])
        sess1.run(train_step1,
                 feed_dict={x1: inputdata, keep_prob1: (1 - DROP_OUT_RATE)})
        if step % 500 == 0:
            print(step, loss1.eval(session=sess1, feed_dict={x1: inputdata, keep_prob1: 1.0}))
            times = [i for i in range(PIXELS)]
            output = y_1.eval(session=sess1, feed_dict={x1: inputdata, keep_prob1: 1.0})
        # if step % 5000 == 0 and step != 0:
        #     times = [i for i in range(PIXELS)]
        #     output = y_1.eval(session=sess1, feed_dict={x1: inputdata, keep_prob1: 1.0})
        #     plt.plot(times, inputdata[0], color='r', lw=2)
        #     plt.plot(times, output[0], color='g', lw=1)
        #     plt.show()
    # save
    learned_W12 = sess1.run(W12)
    learned_b12 = sess1.run(b12)
    learned_W45 = sess1.run(W45)
    learned_b45 = sess1.run(b45)
    numpy.save('../npy/result_W12.npy', learned_W12)
    numpy.save('../npy/result_b12.npy', learned_b12)
    numpy.save('../npy/result_W45.npy', learned_W45)
    numpy.save('../npy/result_b45.npy', learned_b45)

    keep_prob2 = tf.placeholder("float", name='keep_prob2')

    keep_W12 = tf.constant(learned_W12,shape=(PIXELS, H1), dtype=tf.float32, name='keep_W12')
    keep_b12 = tf.constant(learned_b12,shape=[H1], dtype=tf.float32, name='keep_b12')
    keep_W45 = tf.constant(learned_W45,shape=(H1, PIXELS), dtype=tf.float32, name='keep_W45')
    keep_b45 = tf.constant(learned_b45,shape=[PIXELS], dtype=tf.float32, name='keep_b45')

    # create network (2nd)
    x2 = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x2')

    # h12_2 = tf.nn.softsign(tf.matmul(x2, keep_W12) + keep_b12)
    h12_2 = tf.matmul(x2, keep_W12) + keep_b12

    W23 = weight_variable((H1, H2), 'W23')
    b23 = bias_variable([H2], 'b23')
    h23 = tf.nn.softsign(tf.matmul(h12_2, W23) + b23)
    h_drop23 = tf.nn.dropout(h23, keep_prob2)

    W34 = tf.transpose(W23)  # 転置
    b34 = bias_variable([H1], 'b12')
    h34 = tf.nn.softsign(tf.matmul(h_drop23, W34) + b34)

    y_2 = tf.nn.relu(tf.matmul(h34, keep_W45) + keep_b45)
    loss2 = tf.reduce_mean(tf.square(y_2 - x2) * 10000)

    train_step2 = tf.train.AdamOptimizer().minimize(loss2)

    init = tf.initialize_all_variables()
    sess2 = tf.Session()
    sess2.run(init)

    # second learn
    for step in range(10001):
        data_index = random.randint(0,10099)
        inputdata = numpy.array([st[data_index]])
        feed_dict = {
            x2: inputdata,
            keep_prob2: (1 - DROP_OUT_RATE)
        }
        sess2.run(train_step2, feed_dict=feed_dict)
        if step % 500 == 0:
            feed_dict = {
                x2: inputdata,
                keep_prob2: 1.0
            }
            print(step, loss2.eval(session=sess2, feed_dict=feed_dict))
            times = [i for i in range(PIXELS)]
            output = y_2.eval(session=sess2, feed_dict=feed_dict)
    # save
    learned_W23 = sess2.run(W23)
    learned_b23 = sess2.run(b23)
    learned_W34 = sess2.run(W34)
    learned_b34 = sess2.run(b34)
    numpy.save('../npy/result_W23.npy', learned_W23)
    numpy.save('../npy/result_b23.npy', learned_b23)
    numpy.save('../npy/result_W34.npy', learned_W34)
    numpy.save('../npy/result_b34.npy', learned_b34)


    keep_prob3 = tf.placeholder("float", name='keep_prob3')

    keep_W23 = tf.constant(learned_W23,shape=(H1, H2), dtype=tf.float32, name='keep_W23')
    keep_b23 = tf.constant(learned_b23,shape=[H2], dtype=tf.float32, name='keep_b23')
    keep_W34 = tf.constant(learned_W34,shape=(H2, H1), dtype=tf.float32, name='keep_W34')
    keep_b34 = tf.constant(learned_b34,shape=[H1], dtype=tf.float32, name='keep_b34')

    # create network (3rd)
    x3 = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x3')

    # h12_2 = tf.nn.softsign(tf.matmul(x2, keep_W12) + keep_b12)
    h12_3 = tf.matmul(x3, keep_W12) + keep_b12
    h23_3 = tf.nn.softsign(tf.matmul(h12_3, keep_W23) + keep_b23)

    W56 = weight_variable((H2, H3), 'W56')
    b56 = bias_variable([H3], 'b56')
    h56 = tf.nn.softsign(tf.matmul(h23_3, W56) + b56)
    h_drop56 = tf.nn.dropout(h56, keep_prob3)

    W67 = tf.transpose(W56)  # 転置
    b67 = bias_variable([H2], 'b67')
    h67 = tf.nn.softsign(tf.matmul(h_drop56, W67) + b67)

    y_3 = tf.nn.relu(tf.matmul(tf.nn.softsign(tf.matmul(h67, keep_W34) + keep_b34), keep_W45) + keep_b45)
    loss3 = tf.reduce_mean(tf.square(y_3 - x3) * 10000)

    train_step3 = tf.train.AdamOptimizer().minimize(loss3)

    init = tf.initialize_all_variables()
    sess3 = tf.Session()
    sess3.run(init)

    tf.scalar_summary("l2_loss", loss2)
    summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph=sess3.graph)

    # 3rd learn
    for step in range(10001):
        data_index = random.randint(0,10099)
        inputdata = numpy.array([st[data_index]])
        feed_dict = {
            x3: inputdata,
            keep_prob3: 1.0
        }
        sess3.run(train_step3, feed_dict=feed_dict)
        # summary_op = tf.merge_all_summaries()
        # summary_str = sess2.run(summary_op, feed_dict=feed_dict)
        # summary_writer.add_summary(summary_str, step)
        if step % 500 == 0:
            feed_dict = {
                x3: inputdata,
                keep_prob3: 1.0
            }
            print(step, loss3.eval(session=sess3, feed_dict=feed_dict))
            times = [i for i in range(PIXELS)]
            output = y_3.eval(session=sess3, feed_dict=feed_dict)
        if step % 5000 == 0 and step != 0:
            feed_dict = {
                x3: inputdata,
                keep_prob3: 1.0
            }
            times = [i for i in range(PIXELS)]
            output = y_3.eval(session=sess3, feed_dict=feed_dict)
            plt.plot(times, inputdata[0], color='r', lw=2)
            plt.plot(times, output[0], color='g', lw=1)
            plt.show()

    numpy.save('../npy/result_W56.npy', sess3.run(W56))
    numpy.save('../npy/result_b56.npy', sess3.run(b56))
    numpy.save('../npy/result_W67.npy', sess3.run(W67))
    numpy.save('../npy/result_b67.npy', sess3.run(b67))
if __name__ == '__main__':
    main()
