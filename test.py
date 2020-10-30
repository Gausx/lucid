import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('1')

from lucid.misc.io import show, load
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

# Lucid's modelzoo can be accessed as classes in vision_models
import lucid.modelzoo.vision_models as models

# ... or throguh a more systematic factory API
import lucid.modelzoo.nets_factory as nets


print("")
print("Model".ljust(27), " ", "Dataset")
print("")
for name in nets.models_map:
    print(name.ljust(27), " ", nets.models_map[name].dataset)


models.InceptionV4_slim.layers

model = models.InceptionV4_slim()
model.load_graphdef()

# model.show_graph()


model = models.InceptionV4_slim()
model.load_graphdef()

_ = render.render_vis(model, "InceptionV4/InceptionV4/Mixed_6b/concat:0")
