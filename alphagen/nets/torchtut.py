import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#for later: https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier/input


train_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())
print(train_data)
test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())

train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)
for x,y in train_dataloader: 
    print(f"Shape of x [n,c,h,w]: {x.shape}")
    print(f"Shape of y {y.shape} {y.dtype}")
    break
#len(train_data) = 60000 images, but len(train_dataloader) = 938 cus batches)

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#either the CPU or GPU
print(device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #calls constructor of parents class nn.Module
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10)
        )

    def forward(self, x): 
        x = self.flatten(x) #flattens image into 1-d
        logits = self.linear_relu_stack(x) #gets raw scores from neural network, (apply softmax later)
        return logits

model = NeuralNetwork().to(device) #instance of nn class, moves model to device specified above
print(model)

#now, we need a loss function and optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
#for param in model.parameters():

def train(dataloader, model, lossf_n, optimizer): 
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,y) in enumerate(dataloader): 
        x,y = x.to(device), y.to(device)

        #compute pred error
        pred = model(x) #shape (batch_size, numclasses) i.e (64,10)
        loss = loss_fn(pred, y)

        #backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0: 
            loss, cur = loss.item(), (batch+1)*len(x) 
            #no. of batches times data pts in each batch = total processed
            print(f"loss: {loss} [{cur}/{size}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    numbatches = len(dataloader)
    #unlike size (gives u total # of data pts i.e. images)
    #numbatches is datapts/batch_size
    model.eval()
    test_loss, correct = 0,0
    for x,y in dataloader: 
        #iter thru batches, NOT individual data (i.e 900 batches, each with 64  images) 
        #x is tensor (batch_size, 1,28,28), each row = image within batch
        #y is tensor (batch_size,) the label for each image

        x,y = x.to(device), y.to(device)
        test_loss += loss_fn(model(x), y).item()
        predicted_class = model(x).argmax(1)
        correct += (predicted_class == y).type(torch.float).sum().item()
    test_loss /= numbatches
    correct /= size #accuracy 
    print(f"avg test loss: {test_loss}, accuracy: {100*correct} \n")

epochs = 10
for t in range(epochs): 
    print(f"Epoch: {t+1} \n ---")
    train(train_dataloader, model, loss_fn, optimizer)
    #train(test_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print("Done with epoch!")


#saving the model 
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#to recover: define same model w/same architecture (bc it only keeps parameters) 
#model = NeuralNetwork().to(device)
#model.load_state_dict(torch.load("model.pth"), weights_only = True)
#^this load the saved parameters into the model

#make predictions: 












data_iter = iter(train_dataloader)
images, labels = next(data_iter)

def show_images(images, labels):
    plt.figure(figsize=(10, 10))
    num_images = min(len(images), 8)  # Ensure we display at most 8 images
    for i in range(num_images):       # Limit the loop to the grid size
        plt.subplot(2, 4, i + 1)      # Create a 2x4 grid
        plt.imshow(images[i].squeeze(), cmap="gray")  # Squeeze to remove single channel
        plt.title(f"Label: {labels[i].item()}")       # Show the label
        plt.axis("off")
    plt.tight_layout()
    plt.show()

#show_images(images, labels)
#print(type(images), images.shape)

'''
image2 = torch.randn(4,4)
plt.imshow(image2)
plt.show()
'''
