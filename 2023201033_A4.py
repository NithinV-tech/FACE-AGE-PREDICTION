import os
from os.path import join
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import efficientnet_b0


class AgeDataset(Dataset):
    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()
        self.data_path = data_path
        self.annot_path = annot_path
        self.train = train
        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)
    
    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    
    def _transform(self, n_px):
        return Compose([
            Resize(n_px),
            lambda image: self._convert_image_to_rgb(image),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def read_img(self, file_name):
        im_path = join(self.data_path, file_name)
        img = Image.open(im_path)
        img = self.transform(img)
        return img
    
    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img
    
    def __len__(self):
        return len(self.files)

####################################3model############################33
class AgePredictor(nn.Module):
    def __init__(self):
        super(AgePredictor, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        return self.efficientnet(x)
############################################################################


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), ages.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), ages.float())
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')
        test_predictions = predict(test_loader, model)       
        submit['age'] = test_predictions
        submit.to_csv(f'fin_resnet_decay{epoch+1}epochs.csv', index=False)

@torch.no_grad()
def predict(loader, model):
    model.eval()
    predictions = []
    for img in tqdm(loader):
        img = img.to(device)
        pred = model(img)
        predictions.extend(pred.flatten().detach().cpu().tolist())
    return predictions


#########################################driver###################################################################3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_path = '/kaggle/input/sample/content/faces_dataset/train'
train_ann = '/kaggle/input/sample/content/faces_dataset/train.csv'
test_path = '/kaggle/input/sample/content/faces_dataset/test'
test_ann = '/kaggle/input/sample/content/faces_dataset/submission.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)
test_dataset = AgeDataset(test_path, test_ann, train=False)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
submit = pd.read_csv('/kaggle/input/sample/content/faces_dataset/submission.csv')
model = AgePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.05)
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)
preds = predict(test_loader, model)
submit['age'] = preds
submit.to_csv('baseline_efficientnet.csv', index=False)
##########################################################################################################################