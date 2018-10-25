import tensorflow as tf
import matplotlib.pyplot as plt
import time
mnist = tf.keras.datasets.mnist
INDEX = 0

(training_pictures, training_labels), (testing_pictures, testing_labels) = mnist.load_data()  # load data set
model = tf.keras.models.load_model("mnist_model_epoch_5.h5")  # load our model
predictions = model.predict_on_batch(testing_pictures)  # run predictions

for prediction in predictions:  # loop over our predictions
    if 1 not in prediction:  # if the model wasn't 100% sure about a number
        print("\nIndex Number:", INDEX)  # print that number's index
        for index, percentage in enumerate(prediction):  # go over all the probabilities for that number
            print("{}: {:.2f}% chance".format(index, percentage * 100))  # print them out
        print("numbers label:", testing_labels[INDEX])  # print out what the number is supposed to be
        plt.imshow(testing_pictures[INDEX], cmap="gray")  # show the number in question in gray scale
        plt.show()  # display it to the screen
        time.sleep(10)  # wait for 10s before showing the next number (for slide show effect)
    INDEX += 1  # increment the index
