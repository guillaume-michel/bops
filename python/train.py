import time

import numpy as np

import tensorflow as tf

EPOCHS = 5
HIDDEN_NEURONS = 128

# if you want really reproducible results:
# export PYTHONHASHSEED=0 in the calling shell
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(HIDDEN_NEURONS, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start = time.time()
history = model.fit(x_train, y_train, epochs=EPOCHS, shuffle=False)
end = time.time()

results = model.evaluate(x_test, y_test)

print('Train time:', end - start, 's')
