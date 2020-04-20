#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Source: https://www.ibm.com/support/knowledgecenter/SSRU69_1.2.0/base/vision_prepare_custom_deploy.html
"""

import os
import logging as log
import keras
import numpy as np
from PIL import Image

# Import required by Visual Insights
from deploy_interface import DeployCallback

# Import SSD7 model package
import SSD7


class MyDeploy(DeployCallback):
    def __init__(self):
        log.info("CALL MyDeploy.__init__")
        # Define maximum image size (width, height)
        # Images will be NOT resized and must be all of the same size (see load_img method)
        self.img_size = (500, 660)

    def onModelLoading(self, model_path, labels, workspace_path):
        """
        Callback for model loading
        Params:
            model_path: path of the trained model (has been decompressed before this callback)
            workspace_path: recommended temporary workspace
            labels: dict of index -> "category"
        Return: None
        """
        log.info("CALL MyDeploy.onModelLoading")
        self.labels_dict = labels

        model_file = os.path.join(model_path, "model.h5")
        log.info("[model_file] Loading model from %s", model_file)

        # load the Keras model
        self.model = keras.models.load_model(model_file,
                custom_objects = {
                    "AnchorBoxes": SSD7.keras_layer_AnchorBoxes.AnchorBoxes,
                    "compute_loss": SSD7.keras_ssd_loss.SSDLoss().compute_loss
                })
                    

    def onTest(self):
        """
        Return: custom string (used to test API interface)
        """
        log.info("CALL MyDeploy.onTest")
        return "This is Houston. I copy a transmission calling Houston. Over"

    def onInference(self, image_url, params):
        """
        Run inference on a single image
        Params:
            image_url: image path
            params: additional inference options
                "heatmap": for classification, "true" if heatmap requested, "false" if not
        Return:
            if classification: dict
                "label" -> "category", # label name
                "confidence": float, # confidence score between 0 and 1
                "heatmap": "value" # heatmap return [TODO] what does it look like ? No doc provided
            if object detection: list
                "confidence": float # confidence score between 0 and 1
                "label": "category" # label name
                "ymax", "xmax", "xmin", "ymin": coordinates of bounding box
        """
        log.info("CALL MyDeploy.onInference")
        log.info("image_url: %s", image_url)
        # load image to predict, and add dummy batch dimension (as model expect size [batch, width, height, channel])
        image = np.expand_dims(self.load_img(image_url), axis=0)
        # run prediction (a list of one label per image is returned -> first element)
        prediction_encoded = self.model.predict(image)
        prediction = SSD7.decode_detections(prediction_encoded,
                confidence_thresh=0.5,
                img_height=self.img_size[0],
                img_width=self.img_size[1])
        
        return [
            {
                "confidence": box[1],
                "label": self.labels_dict[box[0] - 1],
                "xmin": box[-4],
                "ymin": box[-3],
                "xmax": box[-2],
                "ymax": box[-1]
            } for box in prediction[0]]


    def onFailed(self, deploy_status, e, tb_message):
        """
        Callback for failed deployment
        Params:
            deploy_status: deploy status when failure occurred
            e: Exception object
            tb_message: formatted traceback
        Return: None
        """
        log.info("CALL MyDeploy.onFailed")
        log.error("Deploy status: %s", deploy_status)
        log.error("Traceback message: %s", tb_message)
        log.exception(e)

    def load_img(self, path):
        # given an image path, load and resize the image
        # returns a numpy.array of the shape of the resized image
        img = np.array(Image.open(path), dtype=np.uint8)
        assert(img.shape == self.img_size + (3,))
        return img
