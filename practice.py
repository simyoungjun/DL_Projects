#%
import torch
import os
print(torch.cuda.is_available())
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

# 2번 GPU만 사용하고 싶은 경우 예시(cuda:0에 지정)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"