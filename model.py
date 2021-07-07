import torch.nn as nn
import torch
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_classes=2, classification=False):
        super(LSTM, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Linear(input_size + output_size, output_size)
        self.classification = classification
        if self.classification:
            self.output_dense = nn.Linear(output_size, num_classes)

    def forward(self, input, h_t, c_t):
        combined = torch.cat((input, h_t), 1)

        i = self.sigmoid(self.gate(combined))
        f = self.sigmoid(self.gate(combined))
        c = torch.add(torch.mul(c_t, f), torch.mul(self.tanh(self.gate(combined)), i))
        o = self.sigmoid(self.gate(combined))

        h = torch.mul(self.tanh(c), o)

        if self.classification:
            output = self.output_dense(h)
        else:
            output = h

        return output, h, c

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.cell_size))


class GRU(nn.Module):
    def __init__(self, input_size, output_size, num_classes=2, classification=False):
        super(GRU, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Linear(input_size + output_size, output_size)
        self.classification = classification
        if self.classification:
            self.output_dense = nn.Linear(output_size, num_classes)

    def forward(self, input, h_t):
        combined = torch.cat((input, h_t), 1)

        z = self.sigmoid(self.gate(combined))
        r = self.sigmoid(self.gate(combined))
        h_m = self.tanh(self.gate_x(torch.cat((input, torch.mul(r, h_t)), 1)))
        h = torch.add(torch.mul(z, h_m), torch.mul(1 - z, h_t))

        if self.classification:
            output = self.output_dense(h)
        else:
            output = h

        return output, h

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class RLSTM(nn.Module):
    def __init__(self, input_size, output_size, media_size, num_classes=2, classification=False):
        super(RLSTM, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.media_gate = nn.Linear(media_size, output_size)
        self.input_gate = nn.Linear(input_size + output_size, output_size)
        self.classification = classification
        if self.classification:
            self.output_dense = nn.Linear(output_size, num_classes)

    def forward(self, input, media, h_t, c_t):
        combined = torch.cat((input, h_t), 1)

        i = self.sigmoid(self.input_gate(combined))
        f = self.sigmoid(self.input_gate(combined))
        c = torch.add(torch.mul(c_t, f), torch.mul(self.tanh(self.input_gate(combined)), i))
        o = self.sigmoid(self.input_gate(combined))

        c_r = self.tanh(c)
        c_f = torch.add(c, torch.add(-c_r, torch.mul(c_r, self.sigmoid(self.media_gate(media)))))

        h = torch.mul(self.tanh(c_f), o)

        if self.classification:
            output = self.output_dense(h)
        else:
            output = h

        return output, h, c

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.cell_size))


class RGRU(nn.Module):
    def __init__(self, input_size, output_size, media_size, num_classes=2, classification=False):
        super(RGRU, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.media_gate = nn.Linear(media_size, output_size)
        self.input_gate = nn.Linear(input_size + output_size, output_size)
        self.classification = classification
        if self.classification:
            self.output_dense = nn.Linear(output_size, num_classes)

    def forward(self, input, media, h_t):
        combined = torch.cat((input, h_t), 1)

        z = self.sigmoid(self.gate(combined))
        r = self.sigmoid(self.gate(combined))
        h_m = self.tanh(self.gate_x(torch.cat((input, torch.mul(r, h_t)), 1)))

        h_m_r = self.tanh(h_m)
        h_m_f = torch.add(h_m, torch.add(-h_m_r, torch.mul(h_m_r, self.sigmoid(self.media_gate(media)))))

        h = torch.add(torch.mul(z, h_m_f), torch.mul(1 - z, h_t))

        if self.classification:
            output = self.output_dense(h)
        else:
            output = h

        return output, h

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
