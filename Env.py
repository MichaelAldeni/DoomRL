from vizdoom import DoomGame, GameVariable
import numpy as np
import cv2
from collections import deque

def create_doom_env(frame_skip: int = 4, visible: bool = False):
    game = DoomGame()
    game.load_config("speedrun.cfg")

    if visible:
        game.set_window_visible(True)

    game.init()
    return DoomEnvWrapper(game, frame_skip)

class DoomEnvWrapper:
    def __init__(self, game, frame_skip: int = 4):
        self.game = game
        self.frame_skip = int(frame_skip)
        self.actions = np.eye(8, dtype=np.uint8).tolist()
        
        # Frame Stacking (4 frames)
        self.buffer = deque(maxlen=4)
        self.input_shape = (4, 84, 84)  # Increased resolution to 84x84

        # E1M1 Checkpoints
        self.checkpoints = [
            {'id': 1, 'coords': (1568, -2560, 1664, -2432), 'reward': 100.0},
            {'id': 2, 'coords': (2624, -2560, 2572, -2784), 'reward': 100.0},
            {'id': 3, 'coords': (2912, -4864, 3104, -4672), 'reward': 300.0}
        ]
        self.visited_zones = set()
        
        # State tracking
        self.prev_x = 0
        self.prev_y = 0
        self.health = 100
        self.ammo = 50
        self.kills = 0
        
    def step(self, action_idx):
        vizdoom_reward = self.game.make_action(self.actions[action_idx], self.frame_skip)
        done = self.game.is_episode_finished()
        
        if done:
            # Return stacked state of zeros
            state = np.zeros(self.input_shape, dtype=np.float32)
            if self.game.is_player_dead():
                reward = -5.0
            elif self.game.get_episode_time() >= self.game.get_episode_timeout():
                reward = -2.0 # Timeout penalty
            else:
                reward = 100.0 # Success
            # return last known game vars (normalized) to keep return signature consistent
            try:
                health = float(self.health)
                ammo = float(self.ammo)
            except Exception:
                health = float(self.game.get_game_variable(GameVariable.HEALTH)) if hasattr(self.game, 'get_game_variable') else 0.0
                ammo = float(self.game.get_game_variable(GameVariable.AMMO2)) if hasattr(self.game, 'get_game_variable') else 0.0

            game_vars = np.array([health, ammo], dtype=np.float32)
            game_vars[0] /= 100.0
            game_vars[1] /= 50.0

            return state, game_vars, reward / 50, done
            
        # Get raw frame and process
        frame = self.preprocess(self.game.get_state().screen_buffer)
        self.buffer.append(frame)
        state = np.array(self.buffer, dtype=np.float32).squeeze() # Shape (4, 84, 84)
        
        # --- REWARD SHAPING ---
        curr_x = self.game.get_game_variable(GameVariable.POSITION_X)
        curr_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        reward = 0.0
        
        # 1. REMOVED: Euclidean Distance Reward (It confuses the agent in mazes)
        
        # 2. MOVEMENT REWARD (Slightly reduced)
        travel_dist = np.sqrt((curr_x - self.prev_x)**2 + (curr_y - self.prev_y)**2)
        if travel_dist > 5:
            reward += travel_dist * 0.001
        else:
            # 3. STUCK PENALTY (Crucial for "blocking under wall")
            # If agent attempts to move/turn but distance is near zero
            if action_idx in [0, 1, 3]:
                reward -= 0.1

        # 4. TURNING PENALTY (Discourage excessive turning, but allow it when necessary)
        if action_idx in [5, 6]:
            reward -= 0.005

        # 5. FORWARD BONUS (Encourage forward movement, but keep it small to prevent exploitation)
        if action_idx == 3:
            reward += 0.005

        # 6. COMPASS TO CHECKPOINT 1 (Guidance)
        # Give a small shaping reward when the agent gets closer to checkpoint 1
        if 1 not in self.visited_zones and 2 not in self.visited_zones:
            # approximate target point near the first door/hallway
            dist_to_cp1 = np.sqrt((curr_x - 1568.0)**2 + (curr_y - -2496.0)**2)

            if self.prev_dist_cp1 is None: self.prev_dist_cp1 = dist_to_cp1

            improvement = self.prev_dist_cp1 - dist_to_cp1
            if improvement > 0:
                reward += improvement * 0.001

            self.prev_dist_cp1 = dist_to_cp1
        elif 1 in self.visited_zones and 2 not in self.visited_zones:
            # Guide to Second Door (Exit of large room)
            dist = np.sqrt((curr_x - 2624)**2 + (curr_y - -2560)**2)
                
            if self.prev_dist_cp2 is None: self.prev_dist_cp2 = dist
                
            # We increase the reward slightly here because the room is larger
            improvement = self.prev_dist_cp2 - dist
            if improvement > 0:
                reward += improvement * 0.001
            self.prev_dist_cp2 = dist

        # 7. CHECKPOINTS
        LOITER_PENALTY = -0.05

        for zone in self.checkpoints:
            z_id = zone['id']
            x1, y1, x2, y2 = zone['coords']
            if min(x1, x2) <= curr_x <= max(x1, x2) and min(y1, y2) <= curr_y <= max(y1, y2):
                # First time = Big reward
                if z_id not in self.visited_zones:
                    reward += zone['reward']
                    self.visited_zones.add(z_id)
                    print(f"Reached Checkpoint {z_id}!")
                else:
                # Subsequent visits = Small penalty to encourage progress
                    reward += LOITER_PENALTY

        # 5. AUXILIARY REWARDS (Health/Ammo/Kills)
        # (Same as your original code)
        health = self.game.get_game_variable(GameVariable.HEALTH)
        if self.health > health:
            reward -= (self.health - health) * 0.1 # Reduced penalty to encourage boldness
            self.health = health

        if action_idx == 2: # Shoot action
            ammo = self.game.get_game_variable(GameVariable.AMMO2) # Assuming ammo2 is for the pistol
            if self.ammo > ammo:
                reward -= (self.ammo - ammo) * 0.1 # Small penalty for using ammo, encourages accuracy
                self.ammo = ammo
            
        kill_count = self.game.get_game_variable(GameVariable.KILLCOUNT)
        if kill_count > self.kills:
            reward += 10.0 # Flat reward for kill is usually better than scaling
            self.kills = kill_count
            
        reward += vizdoom_reward
        
        self.prev_x = curr_x
        self.prev_y = curr_y
        
        # package normalized game variables for the usual return signature
        game_vars = np.array([
            float(self.game.get_game_variable(GameVariable.HEALTH)),
            float(self.game.get_game_variable(GameVariable.AMMO2))
        ], dtype=np.float32)
        game_vars[0] /= 100.0
        game_vars[1] /= 50.0

        return state, game_vars, reward / 50, done

    def reset(self):
        self.game.new_episode()
        self.visited_zones.clear()
        self.buffer.clear()
        
        self.prev_x = self.game.get_game_variable(GameVariable.POSITION_X)
        self.prev_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.prev_dist_cp1 = None
        self.prev_dist_cp2 = None   
        
        # Fill buffer with the first frame
        frame = self.preprocess(self.game.get_state().screen_buffer)
        for _ in range(4):
            self.buffer.append(frame)
            
        state = np.array(self.buffer, dtype=np.float32).squeeze()

        game_vars = self.get_normalized_vars()
        
        return state, game_vars

    def preprocess(self, img):
        # Increased resolution to 84x84
        img = cv2.resize(img, (84, 84))
        img = img.astype(np.float32) / 255.0
        return img # Returns (84, 84)
    
    def get_normalized_vars(self):
        # initialize tracked vars for the new episode
        self.health = self.game.get_game_variable(GameVariable.HEALTH)
        self.ammo = self.game.get_game_variable(GameVariable.AMMO2)

        game_vars = np.array([
            float(self.health),
            float(self.ammo)
        ], dtype=np.float32)

        game_vars[0] /= 100.0
        game_vars[1] /= 50.0
 
        return game_vars