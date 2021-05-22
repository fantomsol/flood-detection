import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping
from datatools import DataSplitter
import UNET_MODEL
import remove_noise


def train():
    input_dims = (256, 256, 2)
    # feature_dims = (16, 32, 64, 128, 256)
    feature_dims = (8, 16, 32, 128)
    img_path = r"X:\FLOOD_PREDICT_DATA\train"
    epochs = 20

    data_splitter = DataSplitter(img_path, batch_size=16, validation_split=0.2)
    train_gen, val_gen = data_splitter.getBatchGenerators()

    model = UNET_MODEL.build_model(input_dims, feature_dims, print_summary=False)
    wbce_loss = UNET_MODEL.Weighted_BCE(weight=10)

    mean_iou = MeanIoU(num_classes=2, name="IoU")

    model.compile(optimizer='adam',
                  loss=wbce_loss,
                  metrics=[mean_iou])

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=EarlyStopping(monitor='loss',
                                                verbose=1,
                                                patience=3,
                                                restore_best_weights=1))

    np.save("history_dict.npy", history.history)
    model.save("trained_model")


def img_proc():
    PATH = r"X:\FLOOD_PREDICT_DATA\train"
    remove_noise.remove_noise(PATH)


def model_test():
    MODEL_PATH = "my_model_first.h5"
    IMG_PATH = r"X:\FLOOD_PREDICT_DATA\train"

    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"loss_func": UNET_MODEL.Weighted_BCE})

    dataSplitter = DataSplitter(IMG_PATH, batch_size=1, validation_split=0.1)
    train_gen, val_gen = dataSplitter.getBatchGenerators()

    n_sample = 10
    idxs = np.random.choice(range(train_gen.length), size=n_sample, replace=False)

    loss_func = UNET_MODEL.Weighted_BCE(weight=20)

    for idx in idxs:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        sample = train_gen.__getitem__(idx)

        input_model = sample[0]
        input_img = input_model[0, :, :, 0]  # first channel only
        y_true = sample[1]
        y_pred = model(input_model)

        loss = loss_func(y_true, y_pred).numpy()

        # # compare loss functions
        # loss2 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # loss2 = np.sum(loss2, -1)
        # loss2 = np.sum(loss2, -1)
        # print("BCE = {}, WBCE = {}".format(loss2, loss))

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
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.show()


if __name__ == '__main__':
    # img_proc()
    # train()
    temp = np.load("history_dict.npy", allow_pickle = True)
    pause = True
    hej = 1
    # model_test()
