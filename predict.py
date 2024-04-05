#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2024

@author: muhammad shah
"""

import numpy as np
import tensorflow as tf


class dogcat:
    def __init__(self, filename):
        self.filename = filename

    def predictiondogcat(self):
        # load model        
        model = tf.keras.models.load_model('model.h5')

        # summarize model
        # model.summary()
        imagename = self.filename

        test_image = tf.keras.utils.load_img(imagename, target_size=(64, 64))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = tf.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        print(result)
        if result[0][0] == 1:
            prediction = 'dog'
            return [{"image": prediction}]
        else:
            prediction = 'cat'
            return [{"image": prediction}]
