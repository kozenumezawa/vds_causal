# coding: utf-8

# input pair of data and learn
# this program focuses on
# SV(Salinity causes velocity) and VS(Velocity causes Salinity)

import numpy

import tensorflow as tf

def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)


def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)


def main():
    rawdata = numpy.load('ocean.normalized.npy')
    data = numpy.zeros((rawdata.shape[0], rawdata.shape[2] * 2),
                       dtype=numpy.float32)
    for i in range(rawdata.shape[0]):
        data[i, :rawdata.shape[2]] = rawdata[i, 0]          #   S(salinity)
        data[i, rawdata.shape[2]:] = rawdata[i, 2]          #   V(flow velocity)

    PIXELS = data.shape[1]
    H = 25
    #BATCH_SIZE = data.shape[0]
    BATCH_SIZE = 1
    DROP_OUT_RATE = 0.5

    x = tf.placeholder(tf.float32, [BATCH_SIZE, PIXELS], name='x')

    W = weight_variable((PIXELS, H), 'W')
    b1 = bias_variable([H], 'b1')

    h = tf.nn.softsign(tf.matmul(x, W) + b1)
    keep_prob = tf.placeholder("float", name='keep_prob')
    h_drop = tf.nn.dropout(h, keep_prob)

    W2 = tf.transpose(W)  # 転置
    b2 = bias_variable([PIXELS], 'b2')
    # y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)
    y = tf.matmul(h_drop, W2) + b2

    loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

    tf.scalar_summary("l2_loss", loss)

    train_step = tf.train.AdamOptimizer().minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph=sess.graph)

    for step in range(3001):
        sess.run(train_step,
                 feed_dict={x: [data[step]], keep_prob: (1 - DROP_OUT_RATE)})
        summary_op = tf.merge_all_summaries()
        summary_str = sess.run(summary_op,
                               feed_dict={x: [data[step]], keep_prob: 1.0})
        summary_writer.add_summary(summary_str, step)
        if step % 100 == 0:
            print(loss.eval(session=sess,
                            feed_dict={x: [data[step]], keep_prob: 1.0}))

    result = sess.run(W2)
    numpy.save('result.npy', result)

if __name__ == '__main__':
    main()
