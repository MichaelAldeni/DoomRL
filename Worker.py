import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
from A3C import DoomA3C
from ICM import ICM
from Env import create_doom_env

class Worker(mp.Process):
    def __init__(self, global_model, optimizer, global_icm, icm_optimizer, global_ep_counter, lock, name, frame_skip=4):
        super(Worker, self).__init__()
        self.name = name

        # Shared model and optimizer
        self.global_ep = global_ep_counter
        self.global_model = global_model
        self.optimizer = optimizer

        # Shared ICM model and optimizer
        self.global_icm = global_icm
        self.icm_optimizer = icm_optimizer
        self.local_icm = ICM(num_inputs=1, action_space=8)

        self.lock = lock
        self.frame_skip = int(frame_skip)
        
        # WE DO NOT CREATE THE ENV HERE
        # self.env = create_doom_env()  <-- This caused the error!
        
        self.local_model = DoomA3C(num_inputs=1, action_space=8)
        self.gamma = 0.99
        self.t_max = 20

        self.beta = 0.2  # Weight for forward model loss (curiosity)
        self.lambda_icm = 0.1  # Weight for inverse model loss
        self.eta = 0.01  # Scale factor for intrinsic reward
        self.entropy_coef = 0.01  # Entropy coefficient (increased for more exploration)
        self.max_grad_norm = 40.0  # Gradient clipping threshold

    def run(self):
        # Try to set CPU affinity for this worker (optional). Uses psutil if available.
        try:
            import psutil
            cpu_count = os.cpu_count() or 1
            try:
                rank = int(self.name)
            except Exception:
                rank = 0
            core = rank % cpu_count
            p = psutil.Process()
            p.cpu_affinity([core])
        except Exception:
            # psutil not available or affinity not supported on this platform
            pass

        self.env = create_doom_env(self.frame_skip)
        
        # 1. Rename this to 'worker_step' to avoid confusion
        worker_step = 1 
        
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        state = self.env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # 2. Initialize a local episode counter
        episode_step = 0

        while self.global_ep.value <= 5000:
                
            self.local_model.load_state_dict(self.global_model.state_dict())
            self.local_icm.load_state_dict(self.global_icm.state_dict())
            
            hx = hx.detach()
            cx = cx.detach()

            values = []
            log_probs = []
            rewards = []
            entropies = []

            icm_states = []
            icm_next_states = []
            icm_actions_onehot = []
            icm_actions_indices = []

            for _ in range(self.t_max):
                value, logit, (hx, cx) = self.local_model(state, (hx, cx))
                probs = F.softmax(logit, dim=1)
                m = Categorical(probs)
                action = m.sample()
                
                next_state_np, extrinsic_reward, done = self.env.step(action.item())
                next_state = torch.from_numpy(next_state_np).float().unsqueeze(0)

                # ICM intrinsic reward calculation
                action_one_hot = torch.zeros(1, 8)
                action_one_hot[0, action.item()] = 1.0

                with torch.no_grad():
                     real_feat, pred_feat, _ = self.local_icm(state, next_state, action_one_hot)

                # ICM Intrinsic reward
                intrinsic_reward = self.eta * 0.5 * ((pred_feat - real_feat)**2).sum().item()

                # Total reward is extrinsic + intrinsic
                total_reward = extrinsic_reward + intrinsic_reward

                icm_states.append(state)
                icm_next_states.append(next_state)
                icm_actions_onehot.append(action_one_hot)
                icm_actions_indices.append(action)

                values.append(value)
                log_probs.append(m.log_prob(action))
                rewards.append(total_reward)
                entropies.append(m.entropy())
                
                # 3. Increment counters
                worker_step += 1
                episode_step += 1
                
                # 4. FIX: Use 'episode_step' for the timeout check
                # If the agent gets stuck in a loop for 2000 steps, kill the episode.
                if episode_step >= 2000:
                    done = True
                
                if done:
                    with self.lock:
                        self.global_ep.value += 1
                        
                        if self.global_ep.value % 100 == 0:
                            save_path = f"checkpoints/doom_a3c_ep{self.global_ep.value}.pth"
                            torch.save(self.global_model.state_dict(), save_path)
                            print(f"\nSaved model to {save_path}")

                        #if self.global_ep.value % 10 == 0:
                        #    print(f"Episode {self.global_ep.value} | Worker {self.name} | Steps: {episode_step}")
                    
                    state = self.env.reset()
                    state = torch.from_numpy(state).float().unsqueeze(0)
                    hx = torch.zeros(1, 256)
                    cx = torch.zeros(1, 256)
                    
                    # 5. RESET the episode counter!
                    episode_step = 0
                    break
                else:
                    state = next_state
            
            R = torch.zeros(1, 1)
            if not done:
                value, _, _ = self.local_model(state, (hx, cx))
                R = value.detach()

            values.append(R)
            policy_loss = 0
            value_loss = 0
            
            # A3C Loss calculation
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self.gamma * R
                advantage = R - values[i]
                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - (log_probs[i] * advantage.detach())

            # ICM Loss calculation
            batch_state = torch.cat(icm_states)
            batch_next_state = torch.cat(icm_next_states)
            batch_action_onehot = torch.cat(icm_actions_onehot)
            batch_action_indices = torch.cat(icm_actions_indices)

            real_feat, pred_feat, pred_action_logits = self.local_icm(batch_state, batch_next_state, batch_action_onehot)

            # 1. Forward Loss: || phi(s') - pred_phi(s') ||^2
            forward_loss = 0.5 * F.mse_loss(pred_feat, real_feat.detach(), reduction='sum') / len(rewards)

            # 2. Inverse Loss
            inverse_loss = F.cross_entropy(pred_action_logits, batch_action_indices.squeeze(), reduction='mean')
            icm_loss = (self.beta * forward_loss) + ((1 - self.beta) * inverse_loss) * self.lambda_icm
            
            # Backpropagate combined loss
            self.optimizer.zero_grad()
            a3c_loss = policy_loss + 0.5 * value_loss - 0.01 * sum(entropies)
            a3c_loss.backward()
            
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad
                else:
                    global_param._grad = local_param.grad
            self.optimizer.step()

            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            
            for local_param, global_param in zip(self.local_icm.parameters(), self.global_icm.parameters()):
                 if global_param.grad is None:
                    global_param._grad = local_param.grad
                 else:
                    global_param._grad = local_param.grad
            self.icm_optimizer.step()