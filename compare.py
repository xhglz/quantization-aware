import torch
import torch.nn as nn
import torch.quantization

from data.dataset import prepare_data_loaders
from model.helper import run_benchmark

#%%
data_path = '/home/ly/datasets/imagenet_1k'
saved_model_dir = 'weights/'
float_model_file = 'mobilenet_v2-b0353104.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 30

data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)
run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)