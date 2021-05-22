import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D


# custom weighted binary crossentropy on 2d output
# higher weight means higher penalty for getting 1's wrong

class Weighted_BCE(tf.keras.losses.Loss):
    # input dims: [batch_size, img_width, img_height, 1]
    # output dims: [batch_size,]
    def __init__(self, weight=1):
        super(Weighted_BCE, self).__init__()


    def call(self, y_true, y_pred):
        # avoid inf/nan, more stable numerics
        pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # first term when label == 1, scale loss contribution by weight
        loss = self.weight * y_true * K.log(pred) + (1 - y_true) * K.log(1 - pred)

        # remove redundant dimension and sum up 2d array
        loss = K.squeeze(loss, -1)
        loss = K.sum(loss, -1)
        loss = K.sum(loss, -1)

        # return negated value
        return -loss


def build_model(input_size, feature_dims=tuple, print_summary=False):
    # Build the model
    inputs = tf.keras.layers.Input(input_size)
    x = inputs
    skip_connections = []

    # Contraction path
    for feature in feature_dims:
        x = Conv2D(feature, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = Conv2D(feature, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        if feature != feature_dims[-1]:
            skip_connections.append(x)
            x = MaxPooling2D((2, 2))(x)

    # todo: implement conv layers in the bottleneck?

    # Expansion path
    for feature in reversed(feature_dims[:len(feature_dims) - 1]):
        x = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.concatenate([x, skip_connections[-1]])
        del skip_connections[-1]
        x = Conv2D(feature, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = Conv2D(feature, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    if print_summary:
        model.summary()

    return model