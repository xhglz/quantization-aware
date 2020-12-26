import torch
import torch.nn as nn
import torch.quantization

from data.dataset import prepare_data_loaders
from model.helper import load_model, print_size_of_model, evaluate


data_path = '/home/ly/datasets/imagenet_1k'
saved_model_dir = 'weights/'
float_model_file = 'mobilenet_v2-b0353104.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 32
eval_batch_size = 32

data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)

num_eval_batches = 10

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

