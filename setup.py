import torchdrift
import torch
import torchvision
import redisai
import ml2rt
from copy import deepcopy
import numpy as np


# N_train = 20


# class ImageClassifier(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         resnet = torchvision.models.resnet18(pretrained=True)
#         self.feature_extractor = torch.nn.Sequential(torchvision.transforms.Normalize(mean=mean, std=std),
#                                                      resnet)
        
#         self.classifier = deepcopy(resnet.fc)
#         resnet.fc = torch.nn.Identity()
    
#     def forward(self, x):
#         features = self.feature_extractor(x)
#         classes = self.classifier(features)
#         return classes, features
        

# model = ImageClassifier().eval()
# for p in model.parameters():
#     p.requires_grad_(False)



# torchvision.datasets.utils.download_and_extract_archive('https://download.pytorch.org/tutorial/hymenoptera_data.zip', 'data/')
# val_transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(size=256),
#     torchvision.transforms.CenterCrop(size=(224, 224)),
#     torchvision.transforms.ToTensor()])


# ds_train = torchvision.datasets.ImageFolder('./data/hymenoptera_data/train/', transform=val_transform)

# detector = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=True)
# dl_train = torch.utils.data.DataLoader(ds_train, batch_size=N_train, shuffle=True)
# torchdrift.utils.fit(dl_train, model.feature_extractor, detector, num_batches=1)
# sample, _ = next(iter(dl_train))
# sample = feature_extractor(sample)
# traced = torch.jit.trace(detector, sample)
# traced.save('drift_model.pt')
# user_model_traced = torch.jit.trace(model, sample)
# user_model_traced.save('user_model.pt')


# class WrapperDetector(torch.nn.Module):
#     def __init__(self, detector):
#         super().__init__()
#         self.detector = detector
#         self.QUALIFIED_BATCH_SIZE = 20
    
#     def forward(self, state):
#         if state.shape[0] == self.QUALIFIED_BATCH_SIZE:
#             out = self.detector(state)
#         else:
#             out = torch.ones((1, 1))
#         return out

# detector = torch.jit.load('drift_model.pt')
# wrapper = torch.jit.script(WrapperDetector(detector))
# wrapper.save('wrapped_detector.pt')
