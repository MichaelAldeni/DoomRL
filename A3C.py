import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = weight_shape[0] * weight_shape[2] * weight_shape[3]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class A3C(nn.Module):
    def __init__(self, num_inputs, action_space, input_size: int = 42):
        super(A3C, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # Infer feature_size by forwarding a dummy tensor through convs
        with torch.no_grad():
            dummy = torch.zeros(1, num_inputs, input_size, input_size)
            x = F.elu(self.conv1(dummy))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            self.feature_size = x.view(1, -1).size(1)

        self.var_embedding = nn.Linear(2, 32)  # For health and ammo

        # LSTM will receive visual features + embedded game variables
        self.var_emb_size = 32
        self.lstm = nn.LSTMCell(self.feature_size + self.var_emb_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space)
        self.apply(weights_init)
        
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs, game_vars, hidden_state):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, self.feature_size)
        
        var_emb = F.elu(self.var_embedding(game_vars))
        x = torch.cat([x, var_emb], dim=1)

        hx, cx = self.lstm(x, hidden_state)
        value = self.critic_linear(hx)
        logit = self.actor_linear(hx)

        return value, logit, (hx, cx)