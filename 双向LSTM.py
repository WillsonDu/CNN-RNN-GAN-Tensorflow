import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


learning_rate = 0.01
max_samples = 400000
display_size = 10
batch_size = 128

#实际上图的像素列数，每一行作为一个输入，输入到网络中。
n_input = 28
#LSTM cell的展开宽度，对于图像来说，也是图像的行数
#也就是图像按时间步展开是按照行来展开的。
n_step = 28
#LSTM cell个数
n_hidden = 256
n_class = 10


x = tf.placeholder(tf.float32, shape=[None, n_step, n_input])
y = tf.placeholder(tf.float32, shape =[None, n_class])

#这里的参数只是最后的全连接层的参数，调用BasicLSTMCell这个op，参数已经包在内部了，不需要再定义。
Weight = tf.Variable(tf.random_normal([2 * n_hidden, n_class]))   #参数共享力度比cnn还大
bias = tf.Variable(tf.random_normal([n_class]))


def BiRNN(x, weights, biases):
    #[1, 0, 2]只做第阶和第二阶的转置
    x = tf.transpose(x, [1, 0, 2])
    #把转置后的矩阵reshape成n_input列，行数不固定的矩阵。
    #对一个batch的数据来说，实际上有bacth_size*n_step行。
    x = tf.reshape(x, [-1, n_input])  #-1,表示样本数量不固定
    #拆分成n_step组
    x = tf.split(x, n_step)
    #调用现成的BasicLSTMCell，建立两条完全一样，又独立的LSTM结构
    lstm_qx = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    lstm_hx = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    #两个完全一样的LSTM结构输入到static_bidrectional_rnn中，由这个op来管理双向计算过程。
    #操作 (op, Operation) TensorFlow 图中的节点。在 TensorFlow 中，任何创建、操纵或销毁张量的过程都属于操作。例如，矩阵相乘就是一种操作，该操作以两个张量作为输入，并生成一个张量作为输出。
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_qx, lstm_hx, x, dtype = tf.float32)
    #最后来一个全连接层分类预测
    return tf.matmul(outputs[-1], weights) + biases


pred = BiRNN(x, Weight, bias)
#计算损失、优化、精度（老套路）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accurancy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

#run图过程。
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_step, n_input))
        sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
        if step % display_size == 0:
            acc = sess.run(accurancy, feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(cost, feed_dict = {x:batch_x, y:batch_y})
            print ('Iter' + str(step*batch_size) + ', Minibatch Loss= %.6f'%(loss) + ', Train Accurancy= %.5f'%(acc))

        step += 1
    print ("Optimizer Finished!")


    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape(-1, n_step, n_input)
    test_label = mnist.test.labels[:test_len]
    print ('Testing Accurancy:%.5f'%(sess.run(accurancy, feed_dict={x: test_data, y:test_label})))


    Coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=Coord)