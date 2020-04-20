#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

class Annotation():
    def __init__(self, filename, image_size):
        self.filename = filename
        self.image_size = image_size
        self.objects = []

class Object():
    def __init__(self, label, bbox):
        self.label = label
        self.bbox = bbox



def load_data(images_dir, labels_file, image_size):
    """
    Load data from the 'images_dir' directory. 
    Load labels from the 'labels_file' CSV file.
    """
    # create two dictionnaries
    # * labels: map category name -> id
    # * images: map image path -> category name

    labels_df = pd.read_csv(labels_file)

    images = []
    labels = {"polarbear": 0}
    for image_file in os.listdir(images_dir):
        # create an Annotation (with filename and size)
        img = Annotation(os.path.join(images_dir, image_file), image_size)
        # add labels (Object)
        labels_rows = labels_df[labels_df.image_name == image_file]
        for _, row in labels_rows.iterrows():
            img.objects.append(Object("polarbear", [row.xmin, row.ymin, row.xmax, row.ymax]))
        # add to list of images
        images.append(img)
    return labels, images

