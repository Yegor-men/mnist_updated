import tensorflow as tf

(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

trainX = trainX.reshape(-1, 784)
testX = testX.reshape(-1, 784)

trainX /= 255.0
testX /= 255.0