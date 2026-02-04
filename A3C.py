import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalized_columns_initializer(weights, std=1.0):
    """
    Weights initialization based on the original A3C paper.
    Initializes weights to have a specific standard deviation.
    """
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    """
    Applies the normalization initialization to the network layers.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[0:2]) * weight_shape[3]
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

class DoomA3C(nn.Module):
    def __init__(self, num_inputs, action_space):
        """
        A3C Model for VizDoom.
        
        Args:
            num_inputs (int): Number of input channels (usually 1 for grayscale or 4 for stacked frames).
            action_space (int): Number of discrete actions available in the environment.
        """
        super(DoomA3C, self).__init__()

        # --- 1. Convolutional Feature Extractor ---
        # Architecture follows the "Nature" DQN paper often used in A3C implementations:
        # Conv1: 32 filters, kernel 8x8, stride 4
        # Conv2: 64 filters, kernel 4x4, stride 2
        # Conv3: 64 filters, kernel 3x3, stride 1
        # Input assumes 42x42 or 84x84 resize. This setup works well for 42x42.
        
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # Calculate the flat size after conv layers. 
        # For 42x42 input: (42->21->11->6->3) -> 32 * 3 * 3 = 288
        self.feature_size = 32 * 3 * 3 
        
        # --- 2. LSTM Layer ---
        # The paper uses LSTM to handle partial observability (the agent can't see behind walls).
        self.lstm = nn.LSTMCell(self.feature_size, 256)

        # --- 3. Critic Head (Value Function) ---
        # Estimates V(s): How good is the current state?
        self.critic_linear = nn.Linear(256, 1)

        # --- 4. Actor Head (Policy Function) ---
        # Estimates pi(a|s): Which action should I take?
        self.actor_linear = nn.Linear(256, action_space)

        # --- Initialization ---
        self.apply(weights_init)
        
        # Specific initialization patterns from the A3C paper:
        # Actor output is initialized with a smaller std to ensure initial exploration is uniform (high entropy).
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        
        # Critic output is initialized to 1.0 std
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        
        # LSTM bias initialization (forget gate bias = 1 to remember long-term dependencies initially)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # Put model in train mode
        self.train()

    def forward(self, inputs, hidden_state):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of shape (batch_size, channels, height, width)
            hidden_state: Tuple (hx, cx) from the previous step.
            
        Returns:
            value: Estimated value of the state.
            logit: Raw scores for actions (apply softmax later for probabilities).
            (hx, cx): New hidden states.
        """
        # Feature extraction
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        # Flatten
        x = x.view(-1, self.feature_size)

        # LSTM Step
        # The LSTM requires (hx, cx)
        hx, cx = self.lstm(x, hidden_state)

        # Output Heads
        value = self.critic_linear(hx)
        logit = self.actor_linear(hx)

        return value, logit, (hx, cx)