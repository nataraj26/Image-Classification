# Convolutional Deep Neural Network for Image Classification

## AIM
To design, implement, and evaluate a Convolutional Neural Network (CNN) model for classifying images from the Fashion-MNIST dataset into their respective categories, thereby demonstrating the effectiveness of deep learning techniques in image recognition tasks.

## Problem Statement and Dataset

For this project, the dataset used is the Fashion MNIST dataset, which contains:
Training set: 60,000 grayscale images of size 28×28 pixels.
Test set: 10,000 grayscale images of size 28×28 pixels.
Number of classes: 10
Class labels:
T-shirt/top
Source: The dataset is publicly available through the Keras library and is commonly used for image classification tasks in deep learning.

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/c785e5f3-9533-4bf5-8ae0-68ecc38aa273" />

## DESIGN STEPS

### STEP 1: Define the CNN Architecture

-Import PyTorch libraries.

-Create a class (e.g., FashionCNN) that inherits from nn.Module.

-Add convolutional, pooling, and fully connected layers as per your design.

-Write the forward() function to describe the forward pass.

### STEP 2: Summarize & Visualize the Model

-Use torchsummary.summary(model, (1,28,28)) to check input/output shapes.

-Use torchviz.make_dot() or NN-SVG to generate a visual diagram of the layers.

-Confirm the shapes match your design (e.g., 32@28×28 → 32@14×14, etc.).

### STEP 3: Train & Evaluate the Model

-Load the Fashion-MNIST dataset using torchvision.datasets.FashionMNIST.

-Create DataLoader for batching.

-Define loss function (CrossEntropyLoss) and optimizer (Adam/SGD).

-Train the model for multiple epochs, validate accuracy, and test on unseen data.

-Optionally, plot training loss & accuracy curves.:


## PROGRAM

### Name:
### Register Number:
```python
class CNNClassifier(nn.Module):
  def __init__(self):
    super(CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjusted input features for fc1
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x



```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: Nataraj')
        print('Register Number: 212223230137')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="1535" height="152" alt="image" src="https://github.com/user-attachments/assets/4cf66921-16b9-4a3a-927f-fef7bdb1f1db" />


### Classification Report

<img width="1367" height="482" alt="image" src="https://github.com/user-attachments/assets/a5d091ff-948d-4310-af7d-99272f79a27c" />

### New Sample Data Prediction

<img width="1740" height="380" alt="image" src="https://github.com/user-attachments/assets/7e0d2d11-1a83-4905-b9a7-e857ea4be975" />


## RESULT
After ~3 epochs → ~85–88% accuracy on Fashion-MNIST.
With more epochs & tuning → ~90%+ accuracy.
