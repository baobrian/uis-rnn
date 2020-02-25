# coding= utf-8
import numpy as np
import pandas as pd
import numpy as np
import torch
import tempfile
import uisrnn



# 创建设置模型参数

# construct model
model_args, training_args, inference_args = uisrnn.parse_arguments()
model_args.enable_cuda = True
model_args.rnn_depth = 2
model_args.rnn_hidden_size = 8
model_args.observation_dim = 2
model_args.verbosity = 3
training_args.learning_rate = 0.01
training_args.train_iteration = 2000
training_args.enforce_cluster_id_uniqueness = False
inference_args.test_iteration = 2
model = uisrnn.UISRNN(model_args)



