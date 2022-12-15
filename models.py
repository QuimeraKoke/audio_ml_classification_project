import torch
import torch.nn as nn
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, input_size, sequence_len, bidirect):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi = bidirect
        self.batchNorm = nn.BatchNorm1d(sequence_len)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=0.2,  batch_first=True, bidirectional=bidirect)
        self.fc = nn.Linear(self.hidden_size * 2 if self.bi else self.hidden_size, num_classes)
        
    def forward(self, x):
        hidden = torch.zeros(self.num_layers * 2 if self.bi else self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        x = self.batchNorm(x)
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, input_size, sequence_len, bidirect):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi = bidirect
        self.batchNorm = nn.BatchNorm1d(sequence_len)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirect)
        self.fc = nn.Linear(self.hidden_size * 2 if self.bi else self.hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2 if self.bi else self.num_layers, x.size(0), self.hidden_size).to(DEVICE) 
        c0 = torch.zeros(self.num_layers * 2 if self.bi else self.num_layers, x.size(0), self.hidden_size).to(DEVICE) 
        x = self.batchNorm(x)
        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRU(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, input_size, sequence_len, bidirect):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi = bidirect
        self.batchNorm = nn.BatchNorm1d(sequence_len)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirect)
        self.fc = nn.Linear(self.hidden_size * 2 if self.bi else self.hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2 if self.bi else self.num_layers, x.size(0), self.hidden_size).to(DEVICE) 
        x = self.batchNorm(x)
        out, _ = self.gru(x, h0)  
        out = out[:, -1, :]
         
        out = self.fc(out)
        return out


class CustomLSTM(nn.Module):
    def __init__(self,input_size: int, hidden_size: int, categories: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #it
        self.W_ii = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))

        #ft
        self.W_if = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))

        #gt
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))

        #ot
        self.W_io = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))

        self.out = nn.Linear(hidden_size, categories)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        torch.nn.init.constant_(self.b_if, 1)

    def forward(self,x,init_states=None):

        # Input shape (Batch, seq_len, input_size)
        # Equivalente a usar batch_first = True

        batch, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch,self.hidden_size).to(x.device),
                torch.zeros(batch,self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        
        for t in range(seq_len):
            x_t = x[:,t,:]

            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_ii)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_if)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_ig)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_io)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()

        return self.out(h_t)


class CustomPeepholeLSTM(nn.Module):
    def __init__(self,input_size: int, hidden_size: int, categories: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #it
        self.W_ii = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.W_ci = nn.Parameter(torch.Tensor(hidden_size,hidden_size)) #diferente

        #ft
        self.W_if = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.W_cf = nn.Parameter(torch.Tensor(hidden_size,hidden_size)) #diferente

        #gt
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))

        #ot
        self.W_io = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.W_co = nn.Parameter(torch.Tensor(hidden_size,hidden_size)) #diferente

        self.out = nn.Linear(hidden_size, categories)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        torch.nn.init.constant_(self.b_if, 1)

    def forward(self,x,init_states=None):

        # Input shape (Batch, seq_len, input_size)
        # Equivalente a usar batch_first = True

        batch, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch,self.hidden_size).to(x.device),
                torch.zeros(batch,self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        
        for t in range(seq_len):
            x_t = x[:,t,:]

            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_ii + c_t @ self.W_ci) #diferente
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_if + c_t @ self.W_cf) #diferente
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_ig)
            c_t = f_t * c_t + i_t * g_t
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_io + c_t @ self.W_co) #diferente
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        return self.out(h_t)

class CustomCoupledGateLSTM(nn.Module):
    def __init__(self,input_size: int, hidden_size: int, categories: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #it
        #self.W_ii = nn.Parameter(torch.Tensor(input_size,hidden_size)) #diferencia
        #self.W_hi = nn.Parameter(torch.Tensor(hidden_size,hidden_size)) #diferencia
        #self.b_ii = nn.Parameter(torch.Tensor(hidden_size)) #diferencia

        #Atencion
        #atencion layer
        #softmax
        # avg max pool


        #ft
        self.W_if = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))

        #gt
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))

        #ot
        self.W_io = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))

        self.out = nn.Linear(hidden_size, categories)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        torch.nn.init.constant_(self.b_if, 1)

    def forward(self,x,init_states=None):

        # Input shape (Batch, seq_len, input_size)
        # Equivalente a usar batch_first = True

        batch, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch,self.hidden_size).to(x.device),
                torch.zeros(batch,self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        
        for t in range(seq_len):
            x_t = x[:,t,:]

            # i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_ii) #diferencia
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_if)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_ig)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_io)
            c_t = f_t * c_t + (1 - f_t) * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        return self.out(h_t)

class CustomNoForgetGateLSTM(nn.Module):
    def __init__(self,input_size: int, hidden_size: int, categories: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #it
        self.W_ii = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))

        #ft
        #self.W_if = nn.Parameter(torch.Tensor(input_size,hidden_size)) # diferencia
        #self.W_hf = nn.Parameter(torch.Tensor(hidden_size,hidden_size)) # diferencia
        #self.b_if = nn.Parameter(torch.Tensor(hidden_size)) # diferencia

        #gt
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))

        #ot
        self.W_io = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))

        self.out = nn.Linear(hidden_size, categories)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        #torch.nn.init.constant_(self.b_if, 1) # diferencia

    def forward(self,x,init_states=None):

        # Input shape (Batch, seq_len, input_size)
        # Equivalente a usar batch_first = True

        batch, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch,self.hidden_size).to(x.device),
                torch.zeros(batch,self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        
        for t in range(seq_len):
            x_t = x[:,t,:]

            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_ii)
            #f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_if) # diferencia
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_ig)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_io)
            c_t = c_t + i_t * g_t # diferencia
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        return self.out(h_t)

class CustomNoInputGateLSTM(nn.Module):
    def __init__(self,input_size: int, hidden_size: int, categories: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #it
        #self.W_ii = nn.Parameter(torch.Tensor(input_size,hidden_size)) # diferente
        #self.W_hi = nn.Parameter(torch.Tensor(hidden_size,hidden_size)) # diferente
        #self.b_ii = nn.Parameter(torch.Tensor(hidden_size))

        #ft
        self.W_if = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))

        #gt
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hidden_size)) 
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))

        #ot
        self.W_io = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))

        self.out = nn.Linear(hidden_size, categories)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        torch.nn.init.constant_(self.b_if, 1)

    def forward(self,x,init_states=None):

        # Input shape (Batch, seq_len, input_size)
        # Equivalente a usar batch_first = True

        batch, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch,self.hidden_size).to(x.device),
                torch.zeros(batch,self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        
        for t in range(seq_len):
            x_t = x[:,t,:]

            #i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_ii) # diferente
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_if)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_ig)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_io)
            c_t = f_t * c_t + g_t # diferente
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        return self.out(h_t)

class CustomNoOutputGateLSTM(nn.Module):
    def __init__(self,input_size: int, hidden_size: int, categories: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #it
        self.W_ii = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))

        #ft
        self.W_if = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))

        #gt
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))

        #ot
        #self.W_io = nn.Parameter(torch.Tensor(input_size,hidden_size)) # diferente
        #self.W_ho = nn.Parameter(torch.Tensor(hidden_size,hidden_size)) # diferente
        #self.b_io = nn.Parameter(torch.Tensor(hidden_size)) # diferente

        self.out = nn.Linear(hidden_size, categories)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        torch.nn.init.constant_(self.b_if, 1)

    def forward(self,x,init_states=None):

        # Input shape (Batch, seq_len, input_size)
        # Equivalente a usar batch_first = True

        batch, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch,self.hidden_size).to(x.device),
                torch.zeros(batch,self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        
        for t in range(seq_len):
            x_t = x[:,t,:]

            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_ii)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_if)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_ig)
            #o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_io) # diferente
            c_t = f_t * c_t + i_t * g_t
            h_t = torch.tanh(c_t) # diferente
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        return self.out(h_t)
