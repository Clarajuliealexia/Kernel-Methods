# -*- coding: utf-8 -*-
"""
File creating the final predictions
"""

import utils


labels = [0, 1, 2]

# parameters of the classifier
params = {0:['KS_7', 0.01],
          1:['KS_4', 0.005],
          2:['KS_5', 0.002]}

utils.create_models(labels, params)
