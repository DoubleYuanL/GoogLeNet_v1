import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from inception_v1_utils import *
import cv2

#加载数据集
def load_data():
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = load_dataset()
	return X_train_orig , Y_train_orig , X_test_orig , Y_test_orig 

#初始化数据集 
def init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig ):
	X_train = X_train_orig/255.
	X_test = X_test_orig/255.
	Y_train = convert_to_one_hot(Y_train_orig, 6).T
	Y_test = convert_to_one_hot(Y_test_orig, 6).T
	print ("number of training examples = " + str(X_train.shape[0]))
	print ("number of test examples = " + str(X_test.shape[0])) 
	print ("X_train shape: " + str(X_train.shape)) 
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_test shape: " + str(X_test.shape))
	print ("Y_test shape: " + str(Y_test.shape))
	return X_train, Y_train, X_test, Y_test

#定义模型
def model(X_train,Y_train,X_test,Y_test,learning_rate,num_epochs,minibatch_size,print_cost,isPlot):
	seed = 3
	(m , n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs_1 = []

	X,Y,keep_prob = create_placeholder(n_H0, n_W0, n_C0, n_y)

	Z1 = forward_propagation(X,keep_prob)

	cost_1 = compute_loss(Z1, Y)

	optimizer_1 = tf.train.AdamOptimizer(learning_rate).minimize(cost_1)

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost_1 = 0
			num_minibatches = int(m / minibatch_size) #获取数据块的数量
			#seed = seed + 1
			minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed) 

			#对每个数据块进行处理
			for minibatch in minibatches:
				#选择一个数据块
				(minibatch_X,minibatch_Y) = minibatch
				#最小化这个数据块的成本
				_ , temp_cost_1 = sess.run([optimizer_1,cost_1],feed_dict={X:minibatch_X, Y:minibatch_Y,keep_prob:0.5})

				#累加数据块的成本值
				minibatch_cost_1 += temp_cost_1 / num_minibatches

			#是否打印成本
			if print_cost:
				#每5代打印一次
				if epoch % 10 == 0:
					print("分类器1: 当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost_1))

			#记录成本
			if epoch % 1 == 0:
				costs_1.append(minibatch_cost_1)

		#数据处理完毕，绘制成本曲线
		if isPlot:
			plt.plot(np.squeeze(costs_1))
			plt.ylabel('cost')
			plt.xlabel('iterations (per tens)')
			plt.title("Learning rate =" + str(learning_rate))
			plt.show()
		#保存学习后的参数
		print("参数已经保存到session。")
		saver.save(sess,"model/save_net.ckpt")

		#开始预测数据
		## 计算当前的预测情况
		predict_op = tf.argmax(Z1,1)

		corrent_prediction = tf.equal(predict_op , tf.argmax(Y,1))

		##计算准确度
		accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))
		# print("corrent_prediction accuracy= " + str(accuracy))

		train_accuracy = accuracy.eval({X: X_train, Y: Y_train,keep_prob:1.0})
		test_accuary = accuracy.eval({X: X_test, Y: Y_test,keep_prob:1.0})

		print("训练集准确度：" + str(train_accuracy))
		print("测试集准确度：" + str(test_accuary))

if __name__ == '__main__':
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig  = load_data()
	X_train,Y_train,X_test,Y_test = init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig)
	model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,num_epochs = 100,minibatch_size=64,print_cost=True,isPlot=True)



