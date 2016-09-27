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
    ts = numpy.zeros((rawdata.shape[0], rawdata.shape[2] * 2), dtype=numpy.float32)

    for i in range(rawdata.shape[0]):
        st[i, :rawdata.shape[2]] = rawdata[i, 0]          #   S(salinity)
        st[i, rawdata.shape[2]:] = rawdata[i, 1]          #   T(water temperature)
        ts[i, :rawdata.shape[2]] = rawdata[i, 1]      #   T(salinity)
        ts[i, rawdata.shape[2]:] = rawdata[i, 0]      #   S(water temperature)

    # print(st.shape) = (10100, 424)
    PIXELS = data.shape[1]  # = 424
    H = 70
    BATCH_SIZE = 2
    DROP_OUT_RATE = 0.5

    x = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x')

    W = weight_variable((PIXELS, H), 'W')
    b1 = bias_variable([H], 'b1')

    # h = tf.nn.softsign(tf.matmul(x, W) + b1)
    h = tf.matmul(x, W) + b1
    keep_prob = tf.placeholder("float", name='keep_prob')
    h_drop = tf.nn.dropout(h, keep_prob)

    W2 = tf.transpose(W)  # 転置
    b2 = bias_variable([PIXELS], 'b2')
    y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

    loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

    tf.scalar_summary("l2_loss", loss)

    train_step = tf.train.AdamOptimizer().minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph=sess.graph)

    for step in range(5001):

        inputdata = numpy.array([st[step], ts[step]])
        sess.run(train_step,
                 feed_dict={x: inputdata, keep_prob: (1 - DROP_OUT_RATE)})
        summary_op = tf.merge_all_summaries()
        summary_str = sess.run(summary_op, feed_dict={x: inputdata, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, step)
        if step % 100 == 0:
            print(loss.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0}))
            times = [i for i in range(PIXELS)]
            output = y.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0})
        if step % 1000 == 0:
            times = [i for i in range(PIXELS)]
            output = y.eval(session=sess, feed_dict={x: inputdata, keep_prob: 1.0})
            plt.subplot(2, 1, 1)
            plt.plot(times, inputdata[0], color='r', lw=2)
            plt.plot(times, output[0], color='g', lw=1)
            plt.subplot(2, 1, 2)
            plt.plot(times, inputdata[1], color='r', lw=2)
            plt.plot(times, output[1], color='g', lw=1)
            # plt.show()
    result = sess.run(W2)
    numpy.save('result.npy', result)


    # times = [i for i in range(PIXELS)]
    # Draw Encode/Decode Result
    # for i in range(1):
    #     print(data[0])
    #     plt.plot(times, data[0], color='r', lw=2)
    #     plt.plot(times, data[1], color='g', lw=1)
    # plt.show()
    # plt.savefig("result.png")


if __name__ == '__main__':
    main()
