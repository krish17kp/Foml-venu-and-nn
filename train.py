# make_model.py  (optional demo)
import tensorflow as tf
from tensorflow import keras

(xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
xtr = xtr[..., None]/255.0
xte = xte[..., None]/255.0

model = keras.Sequential([
    keras.layers.Input(shape=(28,28,1)),
    keras.layers.Conv2D(16,3,activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(10,activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(xtr, ytr, epochs=1, batch_size=256, validation_split=0.1)
model.save("mnist_cnn.h5")
print("Saved mnist_cnn.h5")
