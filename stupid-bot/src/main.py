import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from rnn import RNNModel
from data import StupidBotDataset


dataset = StupidBotDataset("../dataset/data.csv")
dataset_size = len(dataset)
dataset_indices = list(range(dataset_size))

batch_size = 1
test_split = int(np.floor(0.2 * dataset_size))  # 20%
# Shuffle dataset indices.
np.random.shuffle(dataset_indices)

train_indices, test_indices = (
    dataset_indices[test_split:],
    dataset_indices[:test_split],
)

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)

model = RNNModel(dataset.unique_characters_length, dataset.unique_characters_length)
model.cuda()

# Define loss and optimizer functions.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the network.
n_epochs = 100
for epoch in range(1, n_epochs + 1):
    for batch_index, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        output, hidden = model(x)  # (16, 21), (1, 1, 32)
        loss = criterion(output, y.view(-1).long())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
            print("Loss: {:.4f}".format(loss.item()))


def predict(model, question):
    """
        Returns the answer to the question.
    """
    question = question.ljust(dataset.longer_question_length)
    question = dataset.text2int(question)
    question = dataset.one_hot_encode(question)
    question = torch.from_numpy(np.array([question])).float().cuda()

    out, hidden = model(question)

    answer = dataset.one_hot_decode(out.cpu())
    answer = dataset.int2text(answer)

    return answer


model.eval()
with torch.no_grad():
    prediction = predict(model, "how are you?")
    print(prediction)
