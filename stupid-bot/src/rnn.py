import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNNModel, self).__init__()

        # Defining some parameters.
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = 32
        self.n_layers = 1

        # region Defining the layers.
        # RNN layer.
        self.rnn = nn.RNN(
            self.input_size,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            nonlinearity="relu",
        )
        # Fully connected layer.
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        # endregion

    def forward(self, x):
        batch_size = x.size(axis=0)

        # Initializing hidden state for first input using method defined below.
        hidden = self.init_hidden(batch_size)  # (1, 1, 32)

        # Passing in the input and hidden state into the model and obtaining outputs.
        out, hidden = self.rnn(
            x, hidden
        )  # input => (1, 16, 21), (1, 1, 32) | output => (1, 16, 32), (1, 1, 32)

        # Reshaping the outputs such that it can be fit into the fully connected layer.
        out = out.contiguous().view(-1, self.hidden_dim)  # (16, 32)
        out = self.fc(out)  # (16, 21)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we will use in the forward pass.
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
