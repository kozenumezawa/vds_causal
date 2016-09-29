# coding: utf-8

# input pair of data and learn
# this program focuses on
# S and T(Salinity causes Temperature) and VS(Velocity causes Salinity)

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)

def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
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
    H2 = 25
    BATCH_SIZE = 1
    DROP_OUT_RATE = 0.5

    x = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x')

    W12 = weight_variable((PIXELS, H1), 'W12')
    b12 = bias_variable([H1], 'b12')
    h12 = tf.nn.sigmoid(tf.matmul(x, W12) + b12)

    keep_prob = tf.placeholder("float", name='keep_prob')
    h_drop = tf.nn.dropout(h12, keep_prob)

    W23 = weight_variable((H1, H2), 'W23')
    b23 = bias_variable([H2], 'b23')
    h23 = tf.nn.sigmoid(tf.matmul(h12, W23) + b23)

    W34 = tf.transpose(W23)  # 転置
    b34 = bias_variable([H1], 'b34')
    h34 = tf.nn.sigmoid(tf.matmul(h23, W34) + b34)

    W45 = tf.transpose(W12)  # 転置
    b45 = bias_variable([PIXELS], 'b45')
    y = tf.nn.relu(tf.matmul(h34, W45) + b45)

    loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

    tf.scalar_summary("l2_loss", loss)

    train_step = tf.train.AdamOptimizer().minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph=sess.graph)

    for step in range(8001):
        inputdata = numpy.array([st[step]])
        sess.run(train_step,
                 feed_dict={x: inputdata, keep_prob: (1 - DROP_OUT_RATE)})
        summary_op = tf.merge_all_summaries()
        summary_str = sess.run(summary_op, feed_dict={x: inputdata, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, step)
        if step % 100 == 0:
            print(loss.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0}))
            times = [i for i in range(PIXELS)]
            output = y.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0})
        if step % 2000 == 0:
            times = [i for i in range(PIXELS)]
            output = y.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0})
            plt.plot(times, inputdata[0], color='r', lw=2)
            plt.plot(times, output[0], color='g', lw=1)
            plt.show()
    result = sess.run(W2)
    numpy.save('result.npy', result)

if __name__ == '__main__':
    main()
