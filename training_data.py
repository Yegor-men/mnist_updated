import tensorflow as tf

(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

trainX = trainX.reshape(-1, 784)
testX = testX.reshape(-1, 784)

trainX = trainX / 255.0
testX = trainY / 255.0