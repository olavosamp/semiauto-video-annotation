import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim

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

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
for elem in params:
    print(elem.size())
# print(params[4].size())

input = torch.randn(1,1,32,32)
output = net(input)
target = torch.randn(84)
target = target.view(1, -1)

criterion = nn.MSELoss()
loss = criterion(output, target)

optimizer = optim.SGD(net.parameters(), lr=0.1)

optimizer.zero_grad()
loss.backward()
optimizer.step()