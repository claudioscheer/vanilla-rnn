import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()

        # Defining some parameters.
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # region Defining the layers.
        # RNN layer.
        self.rnn = nn.RNN(
            input_size, hidden_dim, n_layers, batch_first=True, nonlinearity="relu"
        )
        # Fully connected layer.
        self.fc = nn.Linear(hidden_dim, output_size)
        # endregion

    def forward(self, x):
        batch_size = x.size(axis=0)

        # Initializing hidden state for first input using method defined below.
        hidden = self.init_hidden(batch_size)  # (1, 3, 12)

        # Passing in the input and hidden state into the model and obtaining outputs.
        out, hidden = self.rnn(x, hidden)  # (3, 14, 12), (1, 3, 12)

        # Reshaping the outputs such that it can be fit into the fully connected layer.
        out = out.contiguous().view(-1, self.hidden_dim)  # (42, 12)
        out = self.fc(out)  # (42, 17)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we will use in the forward pass.
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
