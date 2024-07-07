import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
# Step 1: Load and preprocess the dataset
# Load the MNIST dataset, which consists of 60,000 training samples and 10,000 test samples of handwritten digits
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train/255
X_test = X_test/255
X_test.shape



# from tensorflow.keras.optimizers import SGD
# optimizer = SGD(learning_rate=0.01, momentum=0.9)
# model = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax') 
# ])

# model.compile(optimizer = "adam",
#               loss=SparseCategoricalCrossentropy(), 
#               metrics=['accuracy']) 

# history = model.fit(X_train, Y_train, epochs=30,batch_size=1000, validation_data=(X_test, Y_test))

# test_loss, test_accuracy = model.evaluate(X_test, Y_test)

# print(f"Test Accuracy: {test_accuracy}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential([
    # First convolutional layer with 32 filters, kernel size 3x3, ReLU activation
    Input((28, 28,1)),
    Conv2D(32, (3, 3), activation='relu'),
    # Max pooling layer with pool size 2x2
    MaxPooling2D((2, 2)),
    
    # Second convolutional layer with 64 filters, kernel size 3x3, ReLU activation
    Conv2D(64, (3, 3), activation='relu'),
    # Max pooling layer with pool size 2x2
    MaxPooling2D((2, 2)),
    
    # Third convolutional layer with 128 filters, kernel size 3x3, ReLU activation
    Conv2D(128, (3, 3), activation='relu'),
    # Max pooling layer with pool size 2x2
    MaxPooling2D((2, 2)),
    
    # Flatten the output of the convolutional layers
    Flatten(),
    
    # Fully connected (dense) layer with 128 units, ReLU activation
    Dense(128, activation='relu'),
    # Dropout layer to prevent overfitting
    Dropout(0.5),
    
    # Output layer with 26 units (one for each alphabet letter), softmax activation
    Dense(10, activation='softmax')
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=30,batch_size=1000, validation_data=(X_test, Y_test))

test_loss, test_accuracy = model.evaluate(X_test, Y_test)

print(f"Test Accuracy: {test_accuracy}")





data = X_train
prediction = model.predict(data)
prediction


import numpy as np
predicted_classes = np.argmax(prediction, axis=1)



i = np.random.randint(0,60000)
print("Predicted Value: ",predicted_classes[i],"Original value: ",Y_train[i])
plt.gray()
plt.imshow(data[i])
plt.show()