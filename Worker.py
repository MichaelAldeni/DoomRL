import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from A3C import DoomA3C
from Env import create_doom_env

class Worker(mp.Process):
    def __init__(self, global_model, optimizer, global_ep_counter, lock, name, frame_skip=4):
        super(Worker, self).__init__()
        self.name = name
        self.global_ep = global_ep_counter
        self.global_model = global_model
        self.optimizer = optimizer
        self.lock = lock
        # store frame-skip; passed from the main process
        self.frame_skip = int(frame_skip)
        
        # WE DO NOT CREATE THE ENV HERE
        # self.env = create_doom_env()  <-- This caused the error!
        
        self.local_model = DoomA3C(num_inputs=1, action_space=3)
        self.gamma = 0.99
        self.t_max = 20

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
            
            hx = hx.detach()
            cx = cx.detach()

            values = []
            log_probs = []
            rewards = []
            entropies = []

            for _ in range(self.t_max):
                value, logit, (hx, cx) = self.local_model(state, (hx, cx))
                probs = F.softmax(logit, dim=1)
                m = Categorical(probs)
                action = m.sample()
                
                next_state_np, reward, done = self.env.step(action.item())
                
                # 3. Increment counters
                worker_step += 1
                episode_step += 1
                
                values.append(value)
                log_probs.append(m.log_prob(action))
                rewards.append(reward)
                entropies.append(m.entropy())
                
                # 4. FIX: Use 'episode_step' for the timeout check
                # If the agent gets stuck in a loop for 2000 steps, kill the episode.
                if episode_step >= 2000:
                    done = True
                
                if done:
                    with self.lock:
                        self.global_ep.value += 1
                        
                        if self.global_ep.value % 100 == 0:
                            if not os.path.exists('checkpoints'):
                                os.makedirs('checkpoints')
                            save_path = f"checkpoints/doom_a3c_ep{self.global_ep.value}.pth"
                            torch.save(self.global_model.state_dict(), save_path)
                            print(f"Saved model to {save_path}")

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
                    state = torch.from_numpy(next_state_np).float().unsqueeze(0)
            
            R = torch.zeros(1, 1)
            if not done:
                value, _, _ = self.local_model(state, (hx, cx))
                R = value.detach()

            values.append(R)
            policy_loss = 0
            value_loss = 0
            
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self.gamma * R
                advantage = R - values[i]
                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - (log_probs[i] * advantage.detach())
            
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * sum(entropies)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad
                else:
                    global_param._grad = local_param.grad
            
            self.optimizer.step()