import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from datatools import DataSplitter
from utils import purgeNoise
import UNET_MODEL


def main():
    IN_WIDTH = 256
    IN_HEIGHT = 256
    IN_CHANNELS = 2
    INPUT_DIMS = (IN_WIDTH, IN_HEIGHT, IN_CHANNELS)
    FEATURE_DIMS = (16, 32, 64, 128, 256)

    IMG_PATH = r"X:\FLOOD_PREDICT_DATA\train"

    dataSplitter = DataSplitter(IMG_PATH, batch_size=16, validation_split=0.2)
    train_gen, val_gen = dataSplitter.getBatchGenerators()

    model = UNET_MODEL.build_model(INPUT_DIMS, FEATURE_DIMS, print_summary=False)
    loss = UNET_MODEL.Extended_BCE(weight = 20)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=20)
    model.save("my_model.h5")


def img_proc():
    PATH = r"X:\FLOOD_PREDICT_DATA\train"
    purgeNoise(PATH)


def model_test():
    MODEL_PATH = "my_model_first.h5"
    IMG_PATH = r"X:\FLOOD_PREDICT_DATA\train"

    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"loss_func": UNET_MODEL.Extended_BCE})

    dataSplitter = DataSplitter(IMG_PATH, batch_size=1, validation_split=0.1)
    train_gen, val_gen = dataSplitter.getBatchGenerators()

    n_sample = 10
    idxs = np.random.choice(range(train_gen.length), size=n_sample, replace=False)

    loss_func = UNET_MODEL.Extended_BCE(weight=20)

    for idx in idxs:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        sample = train_gen.__getitem__(idx)

        input_model = sample[0]
        input_img = input_model[0, :, :, 0]  # first channel only
        y_true = sample[1]
        y_pred = model(input_model)

        loss = loss_func(y_true, y_pred).numpy()

        y_true_img = y_true[0, :, :, 0]
        y_pred_img = y_pred[0, :, :, 0]
        y_pred_thresh = np.zeros_like(y_pred_img)
        y_pred_thresh[np.where(y_pred_img > 0.5)] = 1

        fig.suptitle("Loss = {}".format(loss))

        plot_imgs = [[input_img, y_pred_img], [y_true_img, y_pred_thresh]]
        plot_titles = [["Input image (vv)", "Model output"], ["True label", "Model prediction"]]

        for row in range(2):
            for col in range(2):
                ax = axs[row][col]
                ax.imshow(plot_imgs[row][col], cmap="gray")
                ax.set_title(plot_titles[row][col])
                ax.tick_params(left=False,
                               bottom=False,
                               labelleft=False,
                               labelbottom=False)

        plt.show()


if __name__ == '__main__':
    # img_proc()
    main()
    #model_test()
