
import tensorflow as tf
from tensorflow.python.keras.metrics import accuracy
def mnist_model_creating(conv2d=64,input_shape=(28, 28, 1),dense=128,output=10):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(conv2d, (3,3), activation='relu', input_shape=input_shape),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(conv2d, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense, activation='relu'),
      tf.keras.layers.Dense(output, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy',"val-accuracy"])

    return model