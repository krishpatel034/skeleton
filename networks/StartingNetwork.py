import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
  def __init__(self):
    # Call nn.Module's constructor--don't forget this
    super().__init__()

    """
    Define layers
    """
    # Explanation of arguments
    # Remember a Convolution layer will take some input volume HxWxC
    # (H = height, W = width, and C = channels) and map it to some output
    # volume H'xW'xC'.
    #
    # Conv2d expects the following arguments
    #   - C, the number of channels in the input
    #   - C', the number of channels in the output
    #   - The filter size (called a kernel size in the documentation)
    #     Below, we specify 5, so our filters will be of size 5x5.
    #   - The amount of padding (default = 0)
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=2)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    # Pooling layer takes two arguments
    #   - Filter size (in this case, 2x2)
    #   - Stride
    self.pool = nn.MaxPool2d(2, 2)

    #same procedure to define fully connected network
    self.fc1 = nn.Linear(32 * 28 * 28, 5)
    self.fc2 = nn.Linear(32, 5)

  def forward(self, x):
    # Comments below give the shape of x
    # n is batch size
    # (batch, channels, height, width)

    # (n, 3, 224, 224)
    x = self.conv1(x)
    x = F.relu(x)
    # (n, 8, 224, 224). notice that the padding made it so we lost no dimensions
    x = self.pool(x)
    # (n, 8, 112, 112). cuts in half. every 2x2 block gets reduced into a 1x1 block b/c of pooling
    x = self.conv2(x)
    x = F.relu(x)
    # (n, 16, 112, 112). once again, notice padding
    x = self.pool(x)
    # (n, 16, 56, 56) 
    x = self.conv3(x)
    x = F.relu(x)
    # (n, 32, 56, 56)
    x = self.pool(x)
    # (n, 32, 28, 28)
    x = torch.reshape(x, (-1, 32 * 28 * 28))
    # (n, 8 * 56 * 56). "A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in input."
    x = self.fc1(x)
    x = F.relu(x)
    # (n, 256)
    #x = self.fc2(x)
    #x = F.relu(x)
    # # (n, 128)
    #x = self.fc3(x)
    # # (n, 10)
    return x