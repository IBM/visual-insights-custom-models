#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, logging, random
# set log level to INFO (same as PAIV)
logging.basicConfig(level=logging.INFO)

import numpy as np
from PIL import Image
from utils import load_data

# ugly hack, but without it, src/train.py can't import src/train_interface.py
# because it should use "from .train_interface ..." and that's not the syntax PAIV expects
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, "..", "src"))

from deploy import MyDeploy
model = MyDeploy()

# specify GPUs used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the directory in which trained model is stored
tmp_dir = os.path.join(path, "tmp")


# load images and pick 10 random ones
labels_dict, images = load_data(os.path.join(path, "..", "..", "datasets", "monkeys-dataset"))
sample_data = random.sample(list(images.items()), 10)

labels_dict_reverse = {v: k for k,v in labels_dict.items()}

# simulate PAIV process (model loading, inference(s), fail if error raised)
try:
    model.onModelLoading(tmp_dir, labels_dict_reverse, os.path.join(path, "tmp"))
    for image_url, label in sample_data:
        logging.info("##### Predicting %s", image_url)
        prediction = model.onInference(image_url, {"heatmap": "false"})
        # print true label and prediction
        logging.info("True label: %s", label)
        logging.info("Prediction: %s", prediction["label"])
except Exception as e:
    model.onFailed("FAILED", e, "ERROR")


