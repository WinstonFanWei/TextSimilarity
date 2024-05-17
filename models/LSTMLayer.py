import torch
import torch.nn as nn

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMLayer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(x, hidden)
        return out, hidden