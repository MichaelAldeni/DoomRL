import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    def __init__(self, num_inputs, action_space, feature_size=288):
        super(ICM, self).__init__()
        self.feature_size = feature_size
        self.action_space = action_space

        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.inverse_fc1 = nn.Linear(self.feature_size * 2, 256)
        self.inverse_fc2 = nn.Linear(256, action_space)

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
        state_feat = self.extract_features(state)
        next_state_feat = self.extract_features(next_state)

        inv_input = torch.cat([state_feat, next_state_feat], dim=1)
        pred_action_logits = self.inverse_fc2(F.elu(self.inverse_fc1(inv_input)))

        fwd_input = torch.cat([state_feat, action_one_hot], dim=1)
        pred_next_state_feat = self.forward_fc2(F.elu(self.forward_fc1(fwd_input)))

        return next_state_feat, pred_next_state_feat, pred_action_logits