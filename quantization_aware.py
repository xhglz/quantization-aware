import time

import torch
import torch.nn as nn
import torch.quantization

from data.dataset import prepare_data_loaders
from model.helper import load_model, accuracy, evaluate, AverageMeter

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return


data_path = '/home/ly/datasets/imagenet_1k'
saved_model_dir = 'weights/'
float_model_file = 'mobilenet_v2-b0353104.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

if __name__ == "__main__":
    train_batch_size = 30
    eval_batch_size = 30

    data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)
    criterion = nn.CrossEntropyLoss()

    #%%
    qat_model = load_model(saved_model_dir + float_model_file)
    qat_model.fuse_model()

    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n', qat_model.features[1].conv)

    num_train_batches = 20
    num_eval_batches = 10
    num_calibration_batches = 10

    # Train and check accuracy after each epoch
    for nepoch in range(8):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            qat_model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates 
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Check the accuracy after each epoch
        quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))


