import requests
import torchvision
import torchvision.transforms as transforms

url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip'
filename = '~/Downloads/imagenet_1k_data.zip'

r = requests.get(url)

with open(filename, 'wb') as f:
    f.write(r.content)
