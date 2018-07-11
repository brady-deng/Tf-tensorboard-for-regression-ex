import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils
#采用tensorflow对一元二次方程进行拟合
#以真实值与预测值的方差作为损失函数



trainsamples = 200  #the number of training samples
testsamples = 60    #the number of testing samples
x = np.linspace(-1,1,trainsamples+testsamples).transpose()      #产生原始数据【-1，1】之间
y = 0.4*pow(x,2)+2*x+np.random.randn(*x.shape)*0.22+0.8     #产生带有噪声的y
def model(x,hidden_weights1,hidden_bias1,ow):       #隐层计算函数
    hidden_layer = tf.nn.sigmoid(tf.matmul(x,hidden_weights1)+hidden_bias1)
    return tf.matmul(hidden_layer,ow)

temp1 = tf.placeholder("float")     #产生一个字符占位符
temp2 = tf.placeholder("float")     #产生一个字符占位符

hw1 = tf.Variable(tf.random_normal([1,10],stddev = 0.01))   #输入层-隐层神经元权重
ow = tf.Variable(tf.random_normal([10,1],stddev = 0.01))    #隐层-输出层神经元权重
b = tf.Variable(tf.random_normal([10],stddev = 0.01))       #偏置

model_y = model(temp1,hw1,b,ow)
cost = tf.pow(model_y-temp2,2)/(2)
train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()    #初始化
    sess.run(init)
    for i in range(1,10):
        trainX,trainY = x[0:trainsamples],y[0:trainsamples]
        for tempv1,tempv2 in zip(trainX,trainY):
            sess.run(train_op,feed_dict={temp1: [[tempv1]],temp2: tempv2})
        testx,testy = x[trainsamples:trainsamples+testsamples],y[0:trainsamples:trainsamples+testsamples]
        cost1 = 0.
        for x1,y1 in zip(testx,testy):
            cost1 += sess.run(cost,feed_dict={temp1:[[x1]],temp2:y1})/testsamples
            print("Average cost for epcch" + str(i) + ":" + str(cost1))
        x,y = sklearn.utils.shuffle(x, y)