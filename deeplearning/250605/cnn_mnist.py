import numpy as np 
from keras.datasets import mnist 
from keras.models import Sequential   
from keras.layers import Dense      
import matplotlib.pyplot as plt

# GPU Settings
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU Available")
else:
    print("GPU Not Available")

# 1. Prepare Training and Test Data

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255


# 2. CNN Classifier Modeling 

from tensorflow import keras
from keras import layers
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
cnn_cls = keras.Model(inputs=inputs, outputs=outputs)

#cnn_cls.summary()

cnn_cls.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

# 3. Train and Evaluate CNN Classifier
history = cnn_cls.fit(train_images, train_labels, epochs=10, batch_size=64)         
test_loss, test_acc = cnn_cls.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.3f}")     
print(f"Test Loss: {test_loss:.3f}")             

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
