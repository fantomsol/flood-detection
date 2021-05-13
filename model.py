import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Lambda, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


img_size = (256, 256)
num_classes = 1

dataset = np.load('data.npz')
y_train = np.load('water_body_label.npy')
x_train = dataset['vv']
#x_val = x_train[-10000:]
#y_val = y_train[-10000:]
#x_train = x_train[:-10000]
#y_train = y_train[:-10000]

x_val = x_train[0:100]
y_val = y_train[0:100]
x_train = x_train[101:300]
y_train = y_train[101:300]
x_train = np.divide(x_train, 255.0)
y_train = np.divide(y_train, 255.0)
x_val = np.divide(x_val, 255.0)
y_val = np.divide(y_val, 255.0)

print("--------------------------------------------------")
print(np.shape(x_train))


def get_model():
    in1 = Input(shape=(256, 256, 1 ))

    conv1 = Conv2D(32, (3, 3), activation= lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(in1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation = lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(conv4)

    up1 = concatenate([UpSampling2D((2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D((2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(conv6)

    up2 = concatenate([UpSampling2D((2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(conv7)
    segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='seg')(conv7)

    model = Model(inputs=[in1], outputs=[segmentation])

    losses = {'seg': 'binary_crossentropy'
            }

    metrics = {'seg': ['acc']
                }
    model.compile(optimizer="adam", loss = losses, metrics=metrics)

    return model
# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


# Build model
model = get_model()
model.summary()

#model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

history = model.fit(
    x_train,
    y_train,
    batch_size=8,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)
