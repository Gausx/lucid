import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('1')

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

# Let's import a model from the Lucid modelzoo!

model = models.InceptionV1()
model.load_graphdef()


# Visualizing a neuron is easy!

_ = render.render_vis(model, "mixed4a_pre_relu:476")
print(1)
