import tensorflow as tf #tf version 2.0.0
import numpy as np
from tensorflow import keras
#Callback is used to stop the training when desired amount of accuracy is achieved
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.92):
            #Must Use metrics=["accuracy"] in compile method
            print("\nReached 92% accuracy so cancelling training!")
            self.model.stop_training = True
            

callbacks = myCallback()
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#Looking at labels and images
import matplotlib.pyplot as plt
plt.imshow(train_images[42])
print(train_labels[42])

#Normalizing Data
train_images=train_images/255
test_images=test_images/255

model=tf.keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(1024,activation=tf.nn.relu),
                          keras.layers.Dense(10,activation=tf.nn.softmax)]) #input_shape is the size of image-28x28
                                                                            #no. of classes=10
model.compile(optimizer=tf.optimizers.Adam(),
             loss=tf.losses.SparseCategoricalCrossentropy(),
             metrics=["accuracy"]) #Search tf.keras.Model, tf.keras.optimizers, tf.keras.losses for more
model.fit(train_images,train_labels,epochs=25,callbacks=[callbacks])

model.evaluate(test_images, test_labels)
