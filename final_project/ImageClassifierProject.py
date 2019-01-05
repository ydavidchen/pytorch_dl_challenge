__title__ = "Final Project through Convolutional Neural Network"
__author__ = "Y. David Chen"
__maintainer__ = "Y. David Chen"
__copyright__ = "Copyright 2018-19"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Development"

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import json

# CONSTANTS
ALEX_NET = models.alexnet(pretrained=True);
BATCH_SIZE = 32;
CROP, RESIZE = 224, 255;
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]);
data_dir = './flower_data';
train_dir = data_dir + '/train';
valid_dir = data_dir + '/valid';

# Load the data
train_transforms = transforms.Compose([transforms.RandomRotation(RESIZE),
                                       transforms.RandomResizedCrop(CROP),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       NORMALIZE,
                                      ]);

test_transforms = transforms.Compose([transforms.Resize(RESIZE),
                                      transforms.CenterCrop(CROP),
                                      transforms.ToTensor(),
                                      NORMALIZE
                                     ]);

train_data = datasets.ImageFolder(train_dir, transform=train_transforms);
test_data = datasets.ImageFolder(valid_dir, transform=test_transforms);

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True);
val_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True);

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f);

# Building the classifier
model = ALEX_NET;
criterion = nn.CrossEntropyLoss();
optimizer = optim.SGD(model.parameters(), lr=0.01);
n_epochs = 2;
saveModel = False;
valid_loss_min = np.Inf;

for epoch in range(1, n_epochs+1):
    train_loss, valid_loss = 0.0, 0.0; #init; used to keep track

    ## Phase I: Training
    model.train();
    for data, target in train_loader:
        optimizer.zero_grad(); #init
        output = model(data);
        loss = criterion(output, target);
        loss.backward();
        optimizer.step();
        train_loss += loss.item()*data.size(0);

    ## Phase II: Validation
    model.eval()
    for data, target in val_loader:
        output = model(data);
        loss = criterion(output, target);
        valid_loss += loss.item()*data.size(0);

    ## Phase III. Save the best model:
    ## Calculate & report average losses
    train_loss = train_loss / len(train_loader.dataset);
    valid_loss = valid_loss / len(val_loader.dataset);

    print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))

# Save the checkpoint
checkpoint = {'state_dict': model.state_dict()};
torch.save(checkpoint, 'classifier.pt');
