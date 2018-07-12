import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils
#采用tensorflow对一元二次方程进行拟合
#以真实值与预测值的方差作为损失函数
#不知道为什么不能把权重和偏置的分布图画出来



trainsamples = 200  #the number of training samples
testsamples = 60    #the number of testing samples
x = np.linspace(-1,1,trainsamples+testsamples).transpose()      #产生原始数据【-1，1】之间
y = 0.4*pow(x,2)+2*x+np.random.randn(*x.shape)*0.22+0.8     #产生带有噪声的y
def model(x,hidden_weights1,hidden_bias1,ow):       #隐层计算函数
    hidden_layer = tf.nn.sigmoid(tf.matmul(x,hidden_weights1)+hidden_bias1)
    return tf.matmul(hidden_layer,ow)
with tf.name_scope('Inputs'):
    temp1 = tf.placeholder("float")     #产生一个字符占位符
    temp2 = tf.placeholder("float")     #产生一个字符占位符
with tf.name_scope('Inference'):
    hw1 = tf.Variable(tf.random_normal([1,10],stddev = 0.01))   #输入层-隐层神经元权重
    ow = tf.Variable(tf.random_normal([10,1],stddev = 0.01))    #隐层-输出层神经元权重
    b = tf.Variable(tf.random_normal([10],stddev = 0.01))       #偏置
    model_y = model(temp1, hw1, b, ow)
    tf.summary.histogram('hw1', hw1)
    tf.summary.histogram('ow', ow)
    tf.summary.histogram('b', b)





with tf.name_scope('cost'):
    cost = tf.pow(model_y-temp2,2)/(2)
    tf.summary.histogram('cost', cost)
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)




sess = tf.Session()
init = tf.global_variables_initializer()    #初始化

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./ex1',sess.graph)
sess.run(init)
for i in range(1,200):
    trainX,trainY = x[0:trainsamples],y[0:trainsamples]
    sess.run(train_op,feed_dict={temp1:trainX,temp2:trainY})
    testx,testy = x[trainsamples:trainsamples+testsamples],y[0:trainsamples:trainsamples+testsamples]
    result = sess.run(merged,feed_dict={temp1: testx, temp2: testy})
    writer.add_summary(result,i)
writer.close()