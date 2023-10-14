import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import ConvNet
from train_functions.starting_train import starting_train


def main():
    csv_file_path = '../data/train.csv'
    image_dir = '../data/train_images'
    validation_split = 0.2

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor,
    ])

    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    initial_dataset = StartingDataset(csv_file=csv_file_path, root_dir=image_dir, transform=transform)
    
    
    total_samples = len(initial_dataset)
    train_size = int(0.95 * total_samples)
    val_size = total_samples - train_size
    train_dataset, test_dataset = random_split(initial_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)


    model = ConvNet()
    
    # Loss function
    # This is the loss you almost always use for classification. Rough formula:
    #   CrossEntropyLoss(y_hat, y) = -Σᵢ yᵢ * log(y_hatᵢ)
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer is used to help us perform gradient descent
    # SGD = Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Use GPU to speed up training
    if torch.cuda.is_available(): # Check if GPU is available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Move the model to the GPU
    model = model.to(device)

    # Training loop

    for epoch in range(constants.EPOCHS): #let's use epoch 10 
        for batch in train_loader: #iterate through the train loader w/ a for loop 
            # images is a 16x1x28x28 tensor containing the 16 images. recall we used batch size 16 way before
            # labels is a 16x1 tensor containing the 16 labels. this is a python tuple unpacking 
            images, labels = batch

            # Move inputs over to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(images) # Same thing as model.forward(images)

            # What shape does outputs have? Answer: (n, 10). 10 possible labels, n images in batch

            # Backprop
            loss = loss_fn(outputs, labels)
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad() # Clear gradients before next iteration
        print('Epoch:', epoch, 'Loss:', loss.item())

    # Change our model to evaluation mode
    # DO NOT FORGET TO DO THIS WHEN EVALUATING
    model.eval()

    # Change model back to train mode
    model.train()

    # Calculate the overall accuracy

    correct = 0
    total = 0
    with torch.no_grad(): # IMPORTANT: turn off gradient computations
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # labels == predictions does an elementwise comparison
            # e.g.                labels = [1, 2, 3, 4]
            #                predictions = [1, 4, 3, 3]
            #      labels == predictions = [1, 0, 1, 0]  (where 1 is true, 0 is false)
            # So the number of correct predictions is the sum of (labels == predictions)
            correct += (labels == predictions).int().sum()
            total += len(predictions)

    print('Accuracy:', (correct / total).item())




if __name__ == "__main__":
    main()
