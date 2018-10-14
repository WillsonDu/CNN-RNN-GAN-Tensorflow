import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import  input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class Net:
    def __init__(self):
        self.x = tf.placeholder(tf.float32,[None,28,28,1])
        self.y = tf.placeholder(tf.float32,[None,10])
        self.conv1_w = tf.Variable(
            tf.random_normal([3,3,1,16],dtype=tf.float32,stddev=0.1))
        self.conv1_b1 = tf.Variable(tf.zeros([16]))
        self.conv2_w = tf.Variable(
            tf.random_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
        self.conv_b2 = tf.Variable(tf.zeros([32]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding="SAME")+self.conv1_b1)
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #14*14

        self.conv2 = tf.nn.relu(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding="SAME") + self.conv_b2)#14*14
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#7*7

        self.flat = tf.reshape(self.pool2,[-1,7*7*32])#-1是剩下的所有。图片的大小和通道

        self.W1 = tf.Variable(tf.random_normal([7*7*32,128],stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.y0 = tf.nn.relu(tf.matmul(self.flat,self.W1)+self.b1)

        self.W2 = tf.Variable(tf.random_normal([128,10],stddev=0.1,dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([10]))

        self.yo = tf.nn.softmax(tf.matmul(self.y0,self.W2)+self.b2)

    def backward(self):
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.yo,labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(0.003).minimize(self.cross_entropy)

        self.corret_prediction = tf.equal(tf.argmax(self.yo,1),tf.argmax(self.y,1))
        self.rst = tf.cast(self.corret_prediction,"float")
        self.accuracy = tf.reduce_mean(self.rst)
if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            xs,ys = mnist.train.next_batch(128)
            batch_xs = xs.reshape([128,28,28,1]) #严格按照NHWC
            _,loss,acc = sess.run([net.optimizer,net.cross_entropy,net.accuracy],feed_dict={net.x:batch_xs,net.y:ys})
            if i % 100==0:
                # print(loss)
                print("精度：{0}".format(acc))