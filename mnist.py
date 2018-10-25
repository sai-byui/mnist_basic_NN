"""
Here we build the model and train it
"""

import tensorflow as tf  # the model building library
mnist = tf.keras.datasets.mnist  # mnist dataset

(training_pictures, training_labels), (testing_pictures, testing_labels) = mnist.load_data()  # load data into training and test sets
training_pictures, testing_pictures = training_pictures / 255.0, testing_pictures / 255.0  # normalize the data

model = tf.keras.models.Sequential()  # tells the model that we're going to start adding layers
model.add(tf.keras.layers.Flatten())  # flatten the input matrix into a vector
model.add(tf.keras.layers.Dense(512, activation="relu"))  # create a dense layer of neurons
model.add(tf.keras.layers.Dropout(0.2))  # add some dropout the neurons to prevent overfitting
model.add(tf.keras.layers.Dense(10, activation="softmax"))  # the final layer is the output layer of neurons
model.compile(optimizer='adam',  # run using an optimized version of gradient descent
              loss='sparse_categorical_crossentropy',  # the kind of loss function we want
              metrics=['accuracy'])  # display the accuracy as we train

model.fit(training_pictures, training_labels, epochs=5)  # fit the data to the model and say now many times to train
model.evaluate(testing_pictures, testing_labels)  # run the model and train it
model.save("/Users/tylercraig/PycharmProjects/AISociety/mnist_model_epoch_5.h5", overwrite=True)  # save it
