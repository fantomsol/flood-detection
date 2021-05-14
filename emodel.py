import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def main():
    # custom weighted binary crossentropy on 2d output
    # higher weight means higher penalty for getting 1's wrong
    def weighted_2d_bce(weight):
        # input dims: [batch_size, img_width, img_height, 1]
        # output dims: [batch_size,]
        def loss_func(true, pred):
            # avoid inf/nan, more stable numerics
            pred = K.clip(pred, K.epsilon(), 1 - K.epsilon())

            # first term when label == 1, scale loss contribution by weight
            loss = weight * true * K.log(pred) + (1 - true) * K.log(1 - pred)

            # remove redundant dimension and sum up 2d array
            loss = K.squeeze(loss, -1)
            loss = K.sum(loss, -1)
            loss = K.sum(loss, -1)

            # return negated value
            return -loss

        return loss_func


    # note: removed dropout between the double-convs. add again if overfitting becomes a problem

    def unet(input_size, class_weight_bias=1, print_summary = False):
        # Build the model
        inputs = tf.keras.layers.Input(input_size)
        # should perhaps be done in preprocessing
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # todo: generalize for parameter search, input feature sizes etc.

        # Contraction path
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=lambda x : tf.nn.leaky_relu(x, alpha=0.1), kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        loss = weighted_2d_bce(class_weight_bias)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        if print_summary:
            model.summary()

        return model

    IN_WIDTH = 256
    IN_HEIGHT = 256
    IN_CHANNELS = 2
    INPUT_DIMS = (IN_WIDTH, IN_HEIGHT, IN_CHANNELS)

    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
    # previous bug: label values 0 or 255 instead of 0 or 1
    # should perhaps be done in preprocessing
    train_y = np.divide(train_y, 255)

    model = unet(INPUT_DIMS, class_weight_bias=20)
    history = model.fit(train_x, train_y, validation_split=0.1, batch_size=8, epochs=5, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()


    test_x = np.load("test_x.npy")
    test_y = np.load("test_y.npy")
    # previous bug: label values 0 or 255 instead of 0 or 1
    # should perhaps be done in preprocessing
    test_y = np.divide(test_y, 255)
    test_x = [test_x]
    test_x = tf.expand_dims(test_x, -1)
    y = model.predict(test_x)

    plt.figure(0)
    plt.gray()
    plt.imshow(y[0])
    plt.show()

    plt.figure(1)
    plt.gray()
    plt.imshow(test_y)
    plt.show()


# sample a number of images/labels from the dataset
# concat vv and vh along channel dimension
def create_subsample(n_samples):
    data = np.load("data.npz", allow_pickle=True)

    vv = data["vv"]
    vh = data["vh"]
    labels = data["flood_label"]

    n = np.size(vv, 0)


    sample_idx = np.random.choice(range(n), n_samples, replace=False)

    sample_vv = vv[sample_idx]
    sample_vh = vh[sample_idx]
    sample_labels = labels[sample_idx]

    vv_test = vv[101]
    vh_test = vh[101]
    y_test = labels[101]

    train_x = np.concatenate((sample_vv, sample_vh), axis=-1)
    test_x = np.concatenate((vv_test,vh_test), axis=-1)

    np.save("train_x", train_x)
    np.save("train_y", sample_labels)
    np.save("test_x", test_x)
    np.save("test_y",y_test)


if __name__ == '__main__':
    create_subsample(7000)
    main()