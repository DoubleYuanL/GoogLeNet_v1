import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from inception_v1_utils import *
import cv2

def predict():
	X,_,keep_prob = create_placeholder(64, 64, 3, 6)

	Z1 = forward_propagation(X,keep_prob)
	Z1 = tf.argmax(Z1,1)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess,tf.train.latest_checkpoint("model/"))

		num = 0
		while (1):
			my_image = "sample/" + str(num) + ".jpg"	
			num_px = 64
			fname =  my_image 
			image = np.array(ndimage.imread(fname, flatten=False))#.astype(np.float32)
			my_predicted_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,64,64,3))/255.
			my_predicted_image = my_predicted_image.astype(np.float32)

			my_predicted_1 = sess.run(Z1, feed_dict={X:my_predicted_image,keep_prob:1.0})
			print(my_predicted_1)
			plt.imshow(image)
			plt.show()
			num = num + 1

if __name__ == '__main__':
	predict()


