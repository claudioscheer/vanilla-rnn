import torch
import torch.nn as nn
import numpy as np
from rnn import RNNModel


text = ["hey how are you", "good i am fine", "have a nice day"]
# Unique characters in sentences.
chars = set("".join(text))
# Maps integer to character.
int2char = dict(enumerate(chars))
# Maps character to integer.
char2int = {char: ind for ind, char in int2char.items()}

# region Padding sentences.
large_sentence_length = len(max(text, key=len))
for i in range(len(text)):
    while len(text[i]) < large_sentence_length:
        text[i] += " "
# endregion

# Maps sentences to input and output sentences.
input_sequences = []  # (3, 14)
target_sequences = []  # (3, 14)

for i in range(len(text)):
    # Remove last character for input sequence.
    input_sequences.append(text[i][:-1])

    # Remove first character for target sequence.
    target_sequences.append(text[i][1:])

# Maps characters to int.
for i in range(len(text)):
    input_sequences[i] = [char2int[character] for character in input_sequences[i]]
    target_sequences[i] = [char2int[character] for character in target_sequences[i]]

# Length of different characters.
unique_characters_size = len(char2int)
# Size of the sequences. -1 because the last character of each sentence is removed.
sequences_length = large_sentence_length - 1
# Size of the batch to train the network.
batch_size = len(text)


def one_hot_encode(
    input_sequences, unique_characters_size, sequences_length, batch_size
):
    # Creating a multi-dimensional array of zeros with the desired output shape.
    features = np.zeros(
        (batch_size, sequences_length, unique_characters_size), dtype=np.float32
    )

    # Replacing the 0 at the relevant character index with a 1 to represent that character.
    for i in range(batch_size):
        for j in range(sequences_length):
            # Each character in the input sequence is represented by a number. Therefore, at the position of the number, we can set as 1.
            features[i, j, input_sequences[i][j]] = 1
    return features


# Input shape = (Batch Size, Sequence Length, One-Hot Encoding Size).
input_sequences = one_hot_encode(
    input_sequences, unique_characters_size, sequences_length, batch_size
)

# region Define the model.
input_sequences = torch.from_numpy(input_sequences).cuda()
target_sequences = torch.Tensor(target_sequences).cuda()

model = RNNModel(
    input_size=unique_characters_size,
    output_size=unique_characters_size,
    hidden_dim=12,
    n_layers=1,
)
model.cuda()

n_epochs = 100

# Define loss and optimizer functions.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the network.
for epoch in range(1, n_epochs + 1):
    # Clears existing gradients from previous epoch.
    optimizer.zero_grad()

    output, hidden = model(input_sequences)
    loss = criterion(output, target_sequences.view(-1).long())

    # Does backpropagation and calculates gradients.
    loss.backward()
    # Updates the weights accordingly.
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
        print("Loss: {:.4f}".format(loss.item()))
# endregion

# region Predicting.
def predict(model, character):
    """
        This function takes in the model and character as arguments and returns the next character prediction and hidden state.
    """
    # One-hot encoding our input to fit into the model.
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, unique_characters_size, character.shape[1], 1)
    character = torch.from_numpy(character).cuda()

    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output.
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden


def sample(model, out_len, start="hey"):
    model.eval()
    start = start.lower()
    # First off, run through the starting characters.
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one.
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return "".join(chars)


output = sample(model, 15, "good")
print(output)
# endregion
