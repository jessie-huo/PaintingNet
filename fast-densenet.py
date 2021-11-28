"""
class1: log <= 2.5      -> 2371 張
class2: 2.5< log <=3.0  -> 7274 張
class3: 3.0< log <=3.5  -> 8073 張
class4: log > 3.5       -> 4666 張

File structure:
root
    |- train
            |- class1
            |- class2
            |- class3
            |- class4
    |- val  
            |- class1
            |- class2
            |- class3
            |- class4
"""


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    start = time.time()
    
    val_acc_history = []
    # model.state_dict(): a dictionary that maps each layer to its parameter tensor.
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'): # gradient calculation enabled only when phase == 'train'
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1) # torch.max() returns the maximum value and its index

                    # backward and update params only when training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # tensor.item(): Returns the value of this tensor as a standard Python number. 
                #                This only works for tensors with one element.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    if model_name == "densenet":
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        # reshape the last layer to output a number of classes specified by us
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()
    return model, input_size


if __name__ == '__main__':

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    data_dir = "./dataset/root"
    model_name = "densenet"
    num_classes = 4
    batch_size = 8
    num_epochs = 15
    feature_extract = True # False: finetune the whole model; True: only change the last layer

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True) # input_size hard-coded as 224
    print('The pretrained densenet:')
    print(model)

    # Data augmentation and normalization for training
    # Normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # Crop a random portion of image and resize it to input_size
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    print('Created data loader ...')

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run.
    # If we are finetuning we will be updating all parameters.
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # Setup the optimizer and loss function (criterion)
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    print('Start feature extract')
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    print('The result densenet:')
    print(model_ft)

    # save the fine-tuned model
    torch.save(model_ft.state_dict(), 'FeatureExtractDenseNet.pt')