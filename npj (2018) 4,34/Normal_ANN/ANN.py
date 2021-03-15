import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np



class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.l1 = nn.Linear(n_input,n_hidden)
        self.l2 = nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x = self.l1(x)
        x = F.relu(x)
        # x = F.dropout(0.5)
        x = self.l2(x)
        # x = F.dropout(0.5)
        x = torch.sigmoid(x)
        return x

feature =  r'F:\Program_Files\Code\Matlab\manhungyung\My_Code\mat\x.mat'
labels = r'F:\Program_Files\Code\Matlab\manhungyung\My_Code\mat\label.mat'

feature = scio.loadmat(feature)
labels = scio.loadmat(labels)
feature = feature['result']
labels = labels['label']
train_x= torch.tensor(feature[:200001,:]).float()
train_labels = torch.tensor(labels[:200001,:]).float()
test_x = torch.tensor(feature[200001:,:]).float()
test_labels = torch.tensor(labels[200001:,:]).float()

net = Net(15,100,1)
net = net.float()
optimizer = torch.optim.Adam(net.parameters(),lr=0.2)
# optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.CrossEntropyLoss()
# print(train_x.data.numpy().dtype)
for epoch in range(100):
    prediction = net.forward(train_x)
    loss = loss_func(prediction, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plt.plot(np.arange(0,100),loss_list)
# plt.show()
#test
prediction = np.squeeze(net.forward(test_x).data.numpy(),axis=1)
prediction = [1 if i > 0.5 else 0 for i in prediction]
prediction = np.array(prediction)
res = np.squeeze(test_labels.data.numpy(),axis=1)
sm = sum(prediction == res)
print(sm/99999)
