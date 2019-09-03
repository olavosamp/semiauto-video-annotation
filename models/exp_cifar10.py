import time
import numpy                    as np
import torch
import torch.nn                 as nn
import torch.nn.functional      as F
import torch.optim              as optim
import torchvision
import torchvision.transforms   as transforms
import matplotlib.pyplot        as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inputChannels = 3

        self.conv1 = nn.Conv2d(self.inputChannels,6,5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*5*5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def imshow(img):
    img = img /2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

if __name__ == "__main__":
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on ", device)

    # Define normalization transform
    transf = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, .5), (.5,.5,.5))]
    )

    # Load data
    datasetFolder = '../datasets/cifar10/'
    trainset = torchvision.datasets.CIFAR10(root=datasetFolder, train=True,
                                            download=True, transform=transf)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=datasetFolder, train=False,
                                        download=True, transform=transf)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Show images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Set up neural network
    net = Net()
    net.to(device)

    crit = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    numEpochs = 6
    print("Starting training for {} epochs.".format(numEpochs))
    start = time.time()
    for epoch in range(numEpochs):
        runningLoss = 0.
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

            if i % 2000 == 1999:
                print("[{:d}, {:5d}] loss: {:.3f}".format(epoch+1, i+1, runningLoss/2000))
                runningLoss = 0.

    trainTime = time.time() - start
    print("Finished Training.\nElapsed time: {:.2f} seconds.\nTime per epoch: {:.2f}".format(
        trainTime, trainTime/numEpochs))

    # Test trained network
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    labels = labels.to(device)

    print("Ground truth: ", " ".join(["{:5s}".format(classes[labels[j]]) for j in range(4)]))
    imshow(torchvision.utils.make_grid(images))

    images = images.to(device)
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    
    print("Prediction: ", " ".join(["{:5s}".format(classes[predicted[j]]) for j in range(4)]))

    # Inference in the whole test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print("Test set size: 10 000 images")
    print("Accuracy: {:.2f}%".format(100*correct/total))

    del dataiter