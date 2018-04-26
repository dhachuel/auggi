########################################################################################################################
## ENVIRONMENT SETUP
########################################################################################################################
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse

# plt.ion()   # interactive mode


########################################################################################################################
## GLOBAL PARAMS
########################################################################################################################
CUDA_AVAILABLE = torch.cuda.is_available()
BATCH_SIZE = 24
CLASSES = 5
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 10
VALIDATION_SIZE = 0.3
SHUFFLE = True
RANDOM_SEED = 1
NUM_WORKERS = 4
PIN_MEMORY = False

########################################################################################################################
## DATA AUGMENTATION
########################################################################################################################
import Augmentor
p = Augmentor.Pipeline()
p.skew(probability=0.5, magnitude=1)
# p.rotate_random_90(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.random_erasing(probability=0.5, rectangle_area=0.25)

########################################################################################################################
## DATA LOADING
########################################################################################################################
data_transforms = {
    'train': transforms.Compose([
        # transforms.Grayscale(),
        # transforms.CenterCrop(256),

        # transforms.RandomGrayscale(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=90),
        # transforms.RandomSizedCrop(),

        # transforms.Resize(256),
        
        # transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 0, 10]),

        
        transforms.Resize(256),
        transforms.CenterCrop(224),
        
        p.torch_transform(),
        

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(),
        # transforms.CenterCrop(256),

        # transforms.RandomGrayscale(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=90),
        # transforms.RandomSizedCrop(),
        
        # transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 0, 10]),
        
        transforms.Resize(256),
        transforms.CenterCrop(224),
        
        p.torch_transform(),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
}


data_dir = "data"

# image_datasets = {
#     x: datasets.ImageFolder(
#         os.path.join(data_dir, x),
#         data_transforms[x]
#     ) for x in ['train', 'val']
# }
# dataloaders = {
#     x: torch.utils.data.DataLoader(
#         image_datasets[x],
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=4
#     ) for x in ['train', 'val']
# }
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes


train_dataset = datasets.ImageFolder(
        os.path.join(data_dir+'/train'), data_transforms['train']
)
val_dataset = datasets.ImageFolder(
        os.path.join(data_dir+'/train'), data_transforms['train']
)



class_names = train_dataset.classes


num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(VALIDATION_SIZE * num_train))

if SHUFFLE:
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

train_idx, val_idx = indices[split:], indices[:split]
dataset_sizes = {
    "train": len(train_idx),
    "val": len(val_idx)
}
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)

dataloaders = {
    "train":train_loader,
    "val":val_loader
}

########################################################################################################################
## DISPLAY SAMPLE
########################################################################################################################

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


########################################################################################################################
## TRAIN FUNCTION
########################################################################################################################
def train_model(model, criterion, optimizer, scheduler, num_epochs=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(">>> phase {}".format(phase))
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iter_count = 0
            for data in dataloaders[phase]:
                print(">>> Loading batch {}...".format(iter_count))
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if CUDA_AVAILABLE:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update iteration counter
                iter_count += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if CUDA_AVAILABLE:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


########################################################################################################################
## TRAIN AND TEST MODELS
########################################################################################################################
# MODEL - FIX ALL WEIGHTS EXCEPT LAST LAYER
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, CLASSES)
if CUDA_AVAILABLE:
    model_conv = model_conv.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1) # lr_scheduler.ReduceLROnPlateau(optimizer_conv, 'min')

model_conv = train_model(
    model=model_conv,
    criterion=criterion,
    optimizer=optimizer_conv,
    scheduler=exp_lr_scheduler,
    num_epochs=EPOCHS
)

# MODEL FT ALL
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, CLASSES)
if CUDA_AVAILABLE:
    model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(
    model=model_ft,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,
    num_epochs=EPOCHS
)


########################################################################################################################
## SAVE MODEL STATES
########################################################################################################################
torch.save(model_conv.state_dict(),"model ft.pt")
torch.save(model_ft.state_dict(),"model_cnn.pt")

model = torch.load("model_conv.pt")
