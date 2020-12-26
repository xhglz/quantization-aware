import torch
import torch.nn as nn
import torch.quantization

from data.dataset import prepare_data_loaders
from model.helper import load_model, print_size_of_model, evaluate

#%%
data_path = '/home/ly/datasets/imagenet_1k'
saved_model_dir = 'weights/'
float_model_file = 'mobilenet_v2-b0353104.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 32
eval_batch_size = 32

data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)
criterion = nn.CrossEntropyLoss()

#%%
num_eval_batches = 10
num_calibration_batches = 10

per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()

# Fuse Conv, bn and relu
per_channel_quantized_model.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', per_channel_quantized_model.features[1].conv)

# Calibrate with the training set
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(per_channel_quantized_model, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',per_channel_quantized_model.features[1].conv)

print("Size of model after quantization")
print_size_of_model(per_channel_quantized_model)

top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)