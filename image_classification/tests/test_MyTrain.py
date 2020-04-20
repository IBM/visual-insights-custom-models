#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, logging
# set log level to INFO (same as PAIV)
logging.basicConfig(level=logging.INFO)

from fakeMonitorHandler import FakeMonitorHandler
from utils import load_data

# ugly hack, but without it, src/train.py can't import src/train_interface.py
# because it should use "from .train_interface ..." and that's not the syntax PAIV expects
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, "..", "src"))

from train import MyTrain
model = MyTrain()

# specify GPUs used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set a directory to store trained model (create it if it does not exist)
tmp_dir = os.path.join(path, "tmp")
os.makedirs(tmp_dir, exist_ok=True)


# simulate data and parameters given by PAIV
labels, images = load_data(os.path.join(path, "..", "..", "datasets", "monkeys-dataset"))
params = {'test_interval': '20', 'learning_rate': '0.001', 'max_iter': '1500', 'weight_decay': '20', 'test_iteration': '100'}

# simulate PAIV process (preprocessing, training, and completed or failed depending on status)
model.onPreprocessing(labels, images, tmp_dir, params)
try:
    model.onTraining(FakeMonitorHandler())
    model.onCompleted(tmp_dir)
except Exception as e:
    model.onFailed("FAILED", e, "ERROR")



