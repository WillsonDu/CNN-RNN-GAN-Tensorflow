import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import  input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class LstmNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])
        #NV变成N(100,28)*V(28)后V的第一层权重：28*128(要变成的权重)
        self.in_w = tf.Variable(tf.truncated_normal(shape=[28,128],stddev=0.1))
        self.in_b = tf.Variable(tf.zeros([128]))
        #输出层神经元权重
        self.out_w = tf.Variable(tf.truncated_normal(shape=[128,10],stddev=0.1))
        self.out_b = tf.Variable(tf.zeros([10]))

    def forward(self):
        #变形合并前的形状是[100,784]->[100,28,28]，
        # 合并后的形状[100*28,28]//NV->NSV->N(NS)V
        y = tf.reshape(self.x,shape=[-1,28])
        #第一层计算后的形状[100*28,128]//N(NS)V
        y = tf.nn.relu(tf.matmul(y,self.in_w)+self.in_b)
        #第一层计算后再变形的形状是从[100*28,128]->[100,28,128]//N(NS)V->NSV
        y = tf.reshape(y,shape=[-1,28,128])
        #记忆细胞神经元的个数（超级参数）
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(128)
        #初始化每一个批次记忆细胞的状态(超级参数)
        init_state = lstm_cell.zero_state(100,dtype=tf.float32)
        #关键的函数
        outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,y,initial_state=init_state,time_major=False)
        #矩阵形状转置，NSV转置成SNV，下标调换，取最后一组(S的最后一步)数据获取的结果
        # y = tf.transpose(outputs,[1,0,2])[-1]
        y = outputs[:,-1,:]
        #输出最终形状是从[100,28,128]->[28,100,128]->[1*100,128]=[100,128]//NSV->N(NS)V->NV
        self.output = tf.nn.softmax(tf.matmul(y,self.out_w)+self.out_b)
    def backward(self):
        self.loss = tf.reduce_mean((self.output-self.y)**2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
if __name__ == '__main__':
    net = LstmNet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000000):
            xs,ys = mnist.train.next_batch(100)
            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs,net.y:ys})
            if i % 100==0:
                test_xs,test_ys = mnist.test.next_batch(100)
                tset_output = sess.run(net.output,feed_dict={net.x:test_xs})

                test_y = np.argmax(test_ys,axis=1)
                test_out = np.argmax(tset_output,axis=1)
                print(np.mean(np.array(test_y==test_out,dtype=np.float32)))