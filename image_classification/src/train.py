#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Source: ttps://www.ibm.com/support/knowledgecenter/SSRU69_1.2.0/base/vision_prepare_custom_train.html
"""

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split
import logging as log
import os

# Import required by Visual Insights
from train_interface import TrainCallback

BATCH_SIZE = 16


class MyTrain(TrainCallback):
    def __init__(self):
        log.info("CALL MyTrain.__init__")
        # Define maximum image size (width, height)
        # Images will be resized (see load_img method)
        self.img_size = (224, 224)

    def onPreprocessing(self, labels, images, workspace_path, params):
        """
        Callback for dataset preprocessing
        Params:
            labels: dict of "category" -> index
            images:
                if classification, dict of "image path" -> "category"
                if detection, list of annotation objects
                (attributes: 'filename', 'size', 'object': list of boxes, each with attrs 'label' and 'bbox')
            workspace_path: recommended temporary workspace
            params: dict of "parameter" -> value
        Return: None
        """
        log.info("CALL MyTrain.onPreprocessing")
        log.info("params: %s", params)
        # store parameters (defined in the GUI when launching training)
        self.params = params

        # Load and preprocess data #####################################################
        # split image paths and labels
        imgs, lbls = zip(*images.items())
        # load all images using their image paths
        X = np.array([self.load_img(img) for img in imgs])
        # replace categories with IDs and one-hot encode them
        y = tf.keras.utils.to_categorical(np.array([labels[lbl_cat] for lbl_cat in lbls]))
        # X and y are both numpy.array objects

        self.num_classes = len(labels)
        self.dataset_size = X.shape[0]
        log.info("images shape = %s", X.shape)
        log.info("labels shape = %s", y.shape)

        # split data -> 20% test / 80% train (using sklearn function)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        # Define custom model architecture #############################################
        # model from https://www.kaggle.com/miljan/image-recognition-testing-popular-cnns
        self.model = tf.keras.models.Sequential()
        self.model.add(BatchNormalization(input_shape=(224, 224, 3)))
        self.model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(BatchNormalization())

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

        # display model architecture summary in logs
        self.model.summary(print_fn=log.info)

    def onTraining(self, monitor_handler):
        """
        Callback for training
        Params:
            monitor_handler: MonitorHandler object for train/test status monitoring
            (see https://www.ibm.com/support/knowledgecenter/SSRU69_1.2.0/base/vision_custom_api.html
                section "Monitoring and reporting statistics")
        Return: None
        """
        log.info("CALL MyTrain.onTraining")

        # function that takes logs (dictionnary containing loss and accuracy values)
        # and calls the monitor_handler methods to update metrics:
        #   * training loss (in updateTrainMetrics)
        #   * testing loss and accuracy (in updateTestMetrics)
        # allowing live graph plot in Visual Insights during training
        def logMetrics(epoch, logs):
            current_iter = (epoch + 1) * self.dataset_size / BATCH_SIZE
            monitor_handler.updateTrainMetrics(
                current_iter,
                int(self.params["max_iter"]),
                logs["loss"],
                epoch + 1)
            monitor_handler.updateTestMetrics(
                current_iter,
                logs["val_acc"],
                logs["val_loss"],
                epoch + 1)

        # launch training using the data we loaded in `onPreprocessing`
        # at the end of each epoch, call the `logMetrics` function as a callback
        # see https://keras.io/callbacks/
        self.model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE,
                       epochs=int(int(self.params["max_iter"]) * BATCH_SIZE / self.dataset_size),
                       validation_data=(self.X_test, self.y_test),
                       callbacks=[LambdaCallback(on_epoch_end=logMetrics)])

    def onCompleted(self, model_path):
        """
        Callback for successful training completion -> used to save model
        Params:
            model_path: absolute model filepath
        Return:
            None
        """
        # if training successful then store the resulting model
        model_file = os.path.join(model_path, "model.h5")
        log.info("CALL MyTrain.onCompleted")
        log.info("[model_file] Saving model to %s", model_file)
        self.model.save(model_file)

    def onFailed(self, train_status, e, tb_message):
        """
        Callback for failed training completion
        Params:
            train_status: training status when failure occurred
            e: Exception object
            tb_message: formatted traceback
        Return: None
        """
        # if training failed then log and raise the error
        log.info("CALL MyTrain.onFailed")
        log.error("Train status: %s", train_status)
        log.error("Traceback message: %s", tb_message)
        log.exception(e)

    def load_img(self, path):
        # given an image path, load and resize the image
        # returns a numpy.array of the shape of the resized image
        return np.array(Image.open(path).resize(self.img_size))
