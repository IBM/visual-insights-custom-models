#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def load_data(path):
    """
    Load data from the 'path' directory
    Data is expected to be sorted by directories representing classes
    (as in the monkeys dataset, used for testing)
    """
    # create two dictionaries
    # * labels: map category name -> id
    # * images: map image path -> category name
    labels = {}
    images = {}
    for i, lbl in enumerate(sorted(os.listdir(path))):
        labels[lbl] = i
        for img in os.listdir(os.path.join(path, lbl)):
            images[os.path.join(path, lbl, img)] = lbl
    return labels, images
