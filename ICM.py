import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    def __init__(self, num_inputs, action_space, feature_size=288):
        super(ICM, self).__init__()
        self.feature_size = feature_size
        self.action_space = action_space

        # 1. Feature Encoder (Same architecture as A3C convs for consistency)
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # 2. Inverse Model: Predict action given s_t and s_{t+1}
        # Input: Concat of two feature vectors (288 * 2)
        self.inverse_fc1 = nn.Linear(self.feature_size * 2, 256)
        self.inverse_fc2 = nn.Linear(256, action_space)

        # 3. Forward Model: Predict phi(s_{t+1}) given phi(s_t) and action
        # Input: Feature vector + Action (usually one-hot)
        self.forward_fc1 = nn.Linear(self.feature_size + action_space, 256)
        self.forward_fc2 = nn.Linear(256, self.feature_size)

    def extract_features(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, self.feature_size)
        return x

    def forward(self, state, next_state, action_one_hot):
        # Get features for current and next state
        state_feat = self.extract_features(state)
        next_state_feat = self.extract_features(next_state)

        # --- Inverse Model ---
        # Concatenate phi(s_t) and phi(s_{t+1})
        inv_input = torch.cat([state_feat, next_state_feat], dim=1)
        pred_action_logits = self.inverse_fc2(F.elu(self.inverse_fc1(inv_input)))

        # --- Forward Model ---
        # Concatenate phi(s_t) and action
        fwd_input = torch.cat([state_feat, action_one_hot], dim=1)
        pred_next_state_feat = self.forward_fc2(F.elu(self.forward_fc1(fwd_input)))

        return next_state_feat, pred_next_state_feat, pred_action_logits