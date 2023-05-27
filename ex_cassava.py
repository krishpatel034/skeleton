import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Define the paths to the dataset and labels
data_dir = '../data/train_images'  # Replace with the path to the dataset directory
train_csv = '../data/train.csv'  # Replace with the path to the train.csv file
test_csv = '../data/test.csv' #Test csv path

# Define the transformation to be applied to the images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a custom dataset for Cassava leaf images
class CassavaLeafDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':
  # Create an instance of the dataset
  dataset = CassavaLeafDataset(data_dir, train_csv, transform=data_transforms)

  # Create a data loader
  batch_size = 32
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

  trainset = CassavaLeafDataset(data_dir, test_csv, transform=data_transforms)
  testloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

  # Load a pre-trained MobileNetV2 model
  model = models.mobilenet_v2(pretrained=True)

  # Modify the last fully connected layer for classification
  num_classes = 5
  num_features = model.classifier[1].in_features
  model.classifier[1] = torch.nn.Linear(num_features, num_classes)

  # Set the device for training
  device = torch.device('mps')

  # Move the model to the device
  model = model.to(device)

  # Define the loss function and optimizer
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  '''
  # Training loop
  image_num = 0
  num_epochs = 10
  for epoch in range(num_epochs):
      running_loss = 0.0
      for inputs, labels in dataloader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          
          optimizer.zero_grad()
          
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          
          loss.backward()
          optimizer.step()
          
          running_loss += loss.item() * inputs.size(0)

          print(f'Still on epoch: {epoch}, image # is: {32 * image_num}')
          image_num += 1
      
      epoch_loss = running_loss / len(dataset)
      print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
  '''

  # Save the trained model
  torch.save(model.state_dict(), 'cassava_model.pth')


  model = TheModelClass(*args, **kwargs)
  model.load_state_dict(torch.load(PATH))
  model.eval()


  correct = 0
  total = 0
  with torch.no_grad(): # IMPORTANT: turn off gradient computations
    for batch in testloader:
      images, labels = batch
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(inputs)
      predictions = torch.argmax(outputs, dim=1)

      # labels == predictions does an elementwise comparison
      # e.g.                labels = [1, 2, 3, 4]
      #                predictions = [1, 4, 3, 3]
      #      labels == predictions = [1, 0, 1, 0]  (where 1 is true, 0 is false)
      # So the number of correct predictions is the sum of (labels == predictions)
      correct += (labels == predictions).int().sum()
      total += len(predictions)

  print('Accuracy:', (correct / total).item())













