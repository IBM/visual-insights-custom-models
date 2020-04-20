#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Source: https://www.ibm.com/support/knowledgecenter/SSRU69_1.2.0/base/vision_prepare_custom_train.html
"""

from PIL import Image
import numpy as np
# import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import logging as log
import os

BATCH_SIZE = 16

# Import required by Visual Insights
from train_interface import TrainCallback

# Import SSD7 model package
import SSD7


class MyTrain(TrainCallback):
    def __init__(self):
        log.info("CALL MyTrain.__init__")
        # Define maximum image size (width, height)
        # Images will NOT be resized and must be all of the same size (see load_img method)
        # If you want to resize them, make sure to update their labels accordingly
        self.img_size = (500, 660)


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
        # store parameters (defined
        self.params = params

        # Define custom model architecture #############################################
        # model from https://github.com/pierluigiferrari/ssd_keras
        self.model = SSD7.build_model(image_size=self.img_size + (3,), n_classes=len(labels))
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(optimizer=adam, loss=SSD7.SSDLoss().compute_loss)
        
        # display model architecture summary in logs
        self.model.summary(print_fn=log.info)


        # Load and preprocess data #####################################################
        X = []
        y = []
        log.info("Loading images...")
        for img in images:
            with Image.open(img.filename) as img_data:
                X.append(np.array(img_data, dtype=np.uint8))
            # class 0 is for background, shift labels of 1
            y.append(np.array([[labels[lbl.label] + 1] + lbl.bbox for lbl in img.objects]))

        # Encode input data for SSD
        predictor_sizes = [self.model.get_layer('classes4').output_shape[1:3],
                           self.model.get_layer('classes5').output_shape[1:3],
                           self.model.get_layer('classes6').output_shape[1:3],
                           self.model.get_layer('classes7').output_shape[1:3]]

        self.ssd_input_encoder = SSD7.SSDInputEncoder(
                img_height=self.img_size[0],
                img_width=self.img_size[1],
                n_classes=len(labels),
                predictor_sizes=predictor_sizes)
 
        X = np.array(X)
        y_encoded = np.array(self.ssd_input_encoder(y))

        # split data -> 20% test / 80% train (using sklearn function)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=0.2)
        self.dataset_size = self.X_train.shape[0]

        log.info("self.X_train.shape = %s", self.X_train.shape)
        log.info("self.X_test.shape  = %s", self.X_test.shape)
        log.info("self.y_train.shape = %s", self.y_train.shape)
        log.info("self.y_test.shape  = %s", self.y_test.shape)



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
        # Note: the SSD7 does not give accuracy, we log 0.0 instead
        # allowing live graph plot in Visual Insights during training
        def logMetrics(epoch, logs):
            current_iter = (epoch + 1) * self.dataset_size / BATCH_SIZE
            monitor_handler.updateTrainMetrics(
                    current_iter,
                    int(self.params["max_iter"]),
                    0.0, # loss_cls
                    logs["loss"], # loss_bbox
                    epoch+1)
            # If you compute an accuracy (mean average precision) on the test set, you can report it here
            # monitor_handler.updateTestMetrics(my_accuracy)

        # launch training using the data we loaded in `onPreprocessing`
        # at the end of each epoch, call the `logMetrics` function as a callback
        # see https://keras.io/callbacks/
        self.model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE, 
                epochs=int(int(self.params["max_iter"]) * BATCH_SIZE / self.dataset_size),
                validation_data=(self.X_test, self.y_test),
                callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=logMetrics)])




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
        # given an image path, load the image
        # returns a numpy.array of the shape of the image
        img = np.array(Image.open(path), dtype=np.uint8)
        assert(img.shape == self.img_size + (3,))
        return img

