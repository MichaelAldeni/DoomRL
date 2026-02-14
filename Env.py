from vizdoom import DoomGame, GameVariable, ScreenResolution
import numpy as np
import cv2
from collections import deque

def create_doom_env(frame_skip: int = 4, visible: bool = False):
    game = DoomGame()
    game.load_config("speedrun.cfg")

    if visible:
        game.set_window_visible(True)
        game.set_screen_resolution(ScreenResolution.RES_640X480)

    game.init()
    return DoomEnvWrapper(game, frame_skip)

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class DoomEnvWrapper:
    def __init__(self, game, frame_skip: int = 4):
        self.game = game
        self.frame_skip = int(frame_skip)
        self.actions = np.eye(7, dtype=np.uint8).tolist()
        
        # Frame Stacking (4 frames)
        self.buffer = deque(maxlen=4)
        self.input_shape = (4, 42, 42)

        # E1M1 Checkpoints
        self.checkpoints = [
            {'id': 1, 'coords': (1568, -2560, 1664, -2432), 'reward': 50.0, 'target': (1616, -2496)},
            {'id': 2, 'coords': (2624, -2560, 2752, -2784), 'reward': 100.0, 'target': (2688, -2672)},
            {'id': 3, 'coords': (2912, -4864, 3104, -4672), 'reward': 200.0, 'target': (3008, -4768)}
        ]
        self.visited_zones = set()
        self.prev_dist_to_goal = None
        
        # State tracking
        self.prev_x = 0
        self.prev_y = 0
        self.health = 100
        self.ammo = 50
        self.kills = 0
        
    def step(self, action_idx):
        self.game.make_action(self.actions[action_idx], self.frame_skip)
        done = self.game.is_episode_finished()
        if done:
            state = np.zeros(self.input_shape, dtype=np.float32)
            reward = 0.0
            for _ in self.visited_zones:
                reward += 10.0
            
            if self.game.is_player_dead():
                reward -= 50.0 # Death penalty
            elif self.game.get_episode_time() >= self.game.get_episode_timeout():
                reward -= 5.0 # Timeout penalty
            else:
                reward += 500.0 # Success
                print("Successfully completed the episode!")
            try:
                health = float(self.health)
                ammo = float(self.ammo)
            except Exception:
                health = float(self.game.get_game_variable(GameVariable.HEALTH)) if hasattr(self.game, 'get_game_variable') else 0.0
                ammo = float(self.game.get_game_variable(GameVariable.AMMO2)) if hasattr(self.game, 'get_game_variable') else 0.0

            game_vars = np.array([health, ammo], dtype=np.float32)
            game_vars[0] /= 100.0
            game_vars[1] /= 50.0

            last_zone = max(self.visited_zones) if self.visited_zones else 0

            return state, game_vars, reward, last_zone, done
            
        frame = self.preprocess(self.game.get_state().screen_buffer)
        self.buffer.append(frame)
        state = np.array(self.buffer, dtype=np.float32).squeeze()
        
        curr_x = self.game.get_game_variable(GameVariable.POSITION_X)
        curr_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        reward = 0.0

        # Timestep reward
        reward += 0.001
        
        # Goal homing rewards based on current zone
        if len(self.visited_zones) == 0:
            # Checkpoint 1
            target_x, target_y = self.checkpoints[0]['target']
            curr_dist = calculate_distance(curr_x, curr_y, target_x, target_y)
            if self.prev_dist_to_goal is not None:
                improvement = self.prev_dist_to_goal - curr_dist
                reward += improvement * 0.02 
            self.prev_dist_to_goal = curr_dist
        elif len(self.visited_zones) == 1:
            # Checkpoint 2
            target_x, target_y = self.checkpoints[1]['target']
            curr_dist = calculate_distance(curr_x, curr_y, target_x, target_y)
            if self.prev_dist_to_goal is not None:
                improvement = self.prev_dist_to_goal - curr_dist
                reward += improvement * 0.02
            self.prev_dist_to_goal = curr_dist
        elif len(self.visited_zones) == 2:
            # Checkpoint 3
            target_x, target_y = self.checkpoints[2]['target']
            curr_dist = calculate_distance(curr_x, curr_y, target_x, target_y)
            if self.prev_dist_to_goal is not None:
                improvement = self.prev_dist_to_goal - curr_dist
                reward += improvement * 0.08 
            self.prev_dist_to_goal = curr_dist
        elif len(self.visited_zones) == 3:
            # Success zone
            target_x, target_y = 2912, -4768
            curr_dist = calculate_distance(curr_x, curr_y, target_x, target_y)
            if self.prev_dist_to_goal is not None:
                improvement = self.prev_dist_to_goal - curr_dist
                reward += improvement * 0.08
            self.prev_dist_to_goal = curr_dist

        # Checkpoints rewards
        for zone in self.checkpoints:
            z_id = zone['id']
            x1, y1, x2, y2 = zone['coords']
            if min(x1, x2) <= curr_x <= max(x1, x2) and min(y1, y2) <= curr_y <= max(y1, y2):
                # First time = Big reward
                if z_id not in self.visited_zones:
                    reward += zone['reward']
                    self.visited_zones.add(z_id)
                    self.prev_dist_to_goal = None  # Reset guidance
                    print(f"Reached Checkpoint {z_id}!")

        # Auxiliary rewards
        health = self.game.get_game_variable(GameVariable.HEALTH)
        if self.health > health:
            reward -= (self.health - health) * 0.05
            self.health = health

        if action_idx == 2:
            ammo = self.game.get_game_variable(GameVariable.AMMO2)
            if self.ammo > ammo:
                reward -= (self.ammo - ammo) * 0.05
                self.ammo = ammo
            
        kill_count = self.game.get_game_variable(GameVariable.KILLCOUNT)
        if kill_count > self.kills:
            reward += 5.0
            self.kills = kill_count
        
        self.prev_x = curr_x
        self.prev_y = curr_y
        
        game_vars = np.array([
            float(self.game.get_game_variable(GameVariable.HEALTH)),
            float(self.game.get_game_variable(GameVariable.AMMO2))
        ], dtype=np.float32)
        game_vars[0] /= 100.0
        game_vars[1] /= 50.0

        return state, game_vars, reward, 0, done

    def reset(self):
        self.game.new_episode()
        self.visited_zones.clear()
        self.buffer.clear()
        
        self.prev_x = self.game.get_game_variable(GameVariable.POSITION_X)
        self.prev_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.prev_dist_to_goal = None   
        
        frame = self.preprocess(self.game.get_state().screen_buffer)
        for _ in range(4):
            self.buffer.append(frame)
            
        state = np.array(self.buffer, dtype=np.float32).squeeze()

        game_vars = self.get_normalized_vars()
        
        return state, game_vars

    def preprocess(self, img):
        img = cv2.resize(img, (42, 42))
        img = img.astype(np.float32) / 255.0
        return img 
    
    def get_normalized_vars(self):
        self.health = self.game.get_game_variable(GameVariable.HEALTH)
        self.ammo = self.game.get_game_variable(GameVariable.AMMO2)

        game_vars = np.array([
            float(self.health),
            float(self.ammo)
        ], dtype=np.float32)

        game_vars[0] /= 100.0
        game_vars[1] /= 50.0
 
        return game_vars