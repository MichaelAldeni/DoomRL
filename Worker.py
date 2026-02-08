import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
from A3C import A3C
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
        
        self.local_model = A3C(num_inputs=1, action_space=8)
        self.gamma = 0.99
        self.t_max = 20

        self.beta = 0.2  # Weight for forward model loss
        self.lambda_icm = 0.1  # Weight for inverse model loss
        self.eta = 0.05  # Scale factor for intrinsic reward
        self.entropy_coef = 0.1  # Entropy coefficient
        self.max_grad_norm = 10.0  # Gradient clipping threshold
        self.reward_clip = 200.0  # Clip per-step rewards to [-clip, +clip]

    def run(self):
        # Set CPU affinity
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
            pass

        self.env = create_doom_env(self.frame_skip)
        
        worker_step = 1 
        
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        state = self.env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        episode_step = 0

        action_counts = [0] * 8

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
                #print(f"Worker {self.name} Step {worker_step} | Action: {action.item()} | Probabilities: {probs.squeeze().detach().cpu().numpy()}")
                
                next_state_np, extrinsic_reward, done = self.env.step(action.item())
                next_state = torch.from_numpy(next_state_np).float().unsqueeze(0)

                # ICM intrinsic reward calculation
                action_one_hot = torch.zeros(1, 8)
                action_one_hot[0, action.item()] = 1.0

                with torch.no_grad():
                     real_feat, pred_feat, _ = self.local_icm(state, next_state, action_one_hot)

                # ICM Intrinsic reward
                intrinsic_reward = self.eta * 0.5 * ((pred_feat - real_feat)**2).sum().item()

                # Total reward is extrinsic + intrinsic, clipped to prevent advantage spikes
                total_reward = max(min(extrinsic_reward + intrinsic_reward, self.reward_clip), -self.reward_clip)

                icm_states.append(state)
                icm_next_states.append(next_state)
                icm_actions_onehot.append(action_one_hot)
                # store action as a 1-d tensor so torch.cat produces a proper target vector
                icm_actions_indices.append(action.unsqueeze(0))

                values.append(value)
                log_probs.append(m.log_prob(action))
                rewards.append(total_reward)
                entropies.append(m.entropy())
                
                worker_step += 1
                episode_step += 1
                action_counts[action.item()] += 1
                
                if done:
                    with self.lock:
                        self.global_ep.value += 1

                        #print(f"\nExtrinsic Reward: {extrinsic_reward:.2f} | Intrinsic Reward: {intrinsic_reward:.2f} | Total Reward: {total_reward:.2f}")
                        
                        if self.global_ep.value % 100 == 0:
                            save_path = f"checkpoints/doom_a3c_ep{self.global_ep.value}.pth"
                            torch.save(self.global_model.state_dict(), save_path)
                            print(f"\nSaved model to {save_path}")
                        
                        # Print action distribution every 50 episodes to detect collapse
                        if self.global_ep.value % 50 == 0:
                            action_names = ["Left", "Right", "Shoot", "Forward", "Back", "TurnL", "TurnR", "Use"]
                            action_dist = {action_names[i]: action_counts[i] for i in range(8)}
                            print(f"\nEp {self.global_ep.value} Worker {self.name} | Steps: {episode_step} | Actions: {action_dist}")
                    
                    state = self.env.reset()
                    state = torch.from_numpy(state).float().unsqueeze(0)
                    hx = torch.zeros(1, 256)
                    cx = torch.zeros(1, 256)
                    
                    episode_step = 0
                    action_counts = [0] * 8
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
            n_steps = len(rewards)
            for i in reversed(range(n_steps)):
                R = rewards[i] + self.gamma * R
                advantage = R - values[i]
                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - (log_probs[i] * advantage.detach())
            
            # Normalize by rollout length so gradient magnitude is consistent
            policy_loss = policy_loss / n_steps
            value_loss = value_loss / n_steps

            # ICM Loss calculation
            batch_state = torch.cat(icm_states)
            batch_next_state = torch.cat(icm_next_states)
            batch_action_onehot = torch.cat(icm_actions_onehot)
            batch_action_indices = torch.cat(icm_actions_indices)

            real_feat, pred_feat, pred_action_logits = self.local_icm(batch_state, batch_next_state, batch_action_onehot)

            # Forward Loss: || phi(s') - pred_phi(s') ||^2
            forward_loss = 0.5 * F.mse_loss(pred_feat, real_feat.detach(), reduction='sum') / len(rewards)

            # Inverse Loss
            inverse_loss = F.cross_entropy(pred_action_logits, batch_action_indices.view(-1), reduction='mean')
            icm_loss = ((self.beta * forward_loss) + ((1 - self.beta) * inverse_loss)) * self.lambda_icm
            
            # Backpropagate combined loss
            self.optimizer.zero_grad()
            entropy_loss = sum(entropies) / len(entropies)
            a3c_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss
            a3c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.max_grad_norm)
            
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad
                else:
                    global_param._grad = local_param.grad
            self.optimizer.step()

            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_icm.parameters(), self.max_grad_norm)
            
            for local_param, global_param in zip(self.local_icm.parameters(), self.global_icm.parameters()):
                 if global_param.grad is None:
                    global_param._grad = local_param.grad
                 else:
                    global_param._grad = local_param.grad
            self.icm_optimizer.step()