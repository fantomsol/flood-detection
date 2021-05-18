import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D


# custom weighted binary crossentropy on 2d output
# higher weight means higher penalty for getting 1's wrong
class Extended_BCE(tf.keras.losses.Loss):
    # input dims: [batch_size, img_width, img_height, 1]
    # output dims: [batch_size,]
    def __init__(self, weight=1):
        super(Extended_BCE, self).__init__()
        self.weight = weight

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

#
# def build_model_ugly_implementation(input_size, class_weight_bias=1, print_summary=False):
#     # Build the model
#     inputs = tf.keras.layers.Input(input_size)
#
#     # Contraction path
#     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
#     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#     p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
#
#     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
#
#     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
#
#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
#
#     c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#     c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
#
#     # Expansive path
#     u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = tf.keras.layers.concatenate([u6, c4])
#     c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#     c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
#
#     u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = tf.keras.layers.concatenate([u7, c3])
#     c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#     c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
#
#     u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = tf.keras.layers.concatenate([u8, c2])
#     c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#     c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
#
#     u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
#     c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#     c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
#
#     outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
#
#     model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#
#     #loss = weighted_2d_bce(class_weight_bias)
#     model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
#
#     if print_summary:
#         model.summary()
#
#     return model
