from vizdoom import DoomGame, GameVariable
import numpy as np
import cv2

def create_doom_env(frame_skip: int = 4):
    game = DoomGame()
    game.load_config("speedrun.cfg")
    game.init()
    return DoomEnvWrapper(game, frame_skip)

class DoomEnvWrapper:
    def __init__(self, game, frame_skip: int = 4):
        self.game = game
        self.frame_skip = int(frame_skip)
        self.actions = np.eye(8, dtype=np.uint8).tolist()

        # E1M1 Exit Switch Coordinates
        self.GOAL_X = 2955.0
        self.GOAL_Y = -4772.0
        
        # Exploration tracking
        self.checkpoints = [
            # The hallway after the first door
            {'id': 1, 'coords': (1568, -2560, 1664, -2432), 'reward': 50.0},
            # The zig-zag room
            {'id': 2, 'coords': (2944, -4160, 3072, -4032), 'reward': 50.0},
            # The area around the exit switch
            {'id': 3, 'coords': (2912, -4864, 3104, -4672), 'reward': 100.0}
        ]
        self.visited_zones = set()
        
        # State tracking for reward calculation
        self.prev_x = 0
        self.prev_y = 0
        self.prev_dist = 0
        
    def _calc_dist(self, x, y):
        return np.sqrt((x - self.GOAL_X)**2 + (y - self.GOAL_Y)**2)

    def step(self, action_idx):
        vizdoom_reward = self.game.make_action(self.actions[action_idx], self.frame_skip)
        self.ticks += self.frame_skip
        done = self.game.is_episode_finished()
        if done:
            state = np.zeros((1, 42, 42), dtype=np.float32)
            if self.game.is_player_dead():
                reward = -5.0
            elif self.ticks >= self.game.get_episode_timeout():
                reward = -2.0
            else:
                reward = 50.0
            
            #print("Reward for episode end:", self.game.get_total_reward())
            return state, reward, done
            
        state = self.preprocess(self.game.get_state().screen_buffer)
        
        curr_x = self.game.get_game_variable(GameVariable.POSITION_X)
        curr_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        curr_dist = self._calc_dist(curr_x, curr_y)
        
        reward = 0.0
        
        # Goal Homing
        #dist_improvement = (self.prev_dist - curr_dist)
        #reward += dist_improvement * 0.01
        
        # Calculate distance traveled this step
        travel_dist = np.sqrt((curr_x - self.prev_x)**2 + (curr_y - self.prev_y)**2)
        if travel_dist > 0.5:  # Filter jitter
            reward += travel_dist * 0.005
        
        # Penalty for acting without moving 
        #if action_idx in (2, 5, 6, 7) and travel_dist < 1.0:
        #    reward -= 0.05
        
        # Checkpoint Rewards
        for zone in self.checkpoints:
            z_id = zone['id']
            if z_id not in self.visited_zones:
                x1, y1, x2, y2 = zone['coords']
                if min(x1, x2) <= curr_x <= max(x1, x2) and min(y1, y2) <= curr_y <= max(y1, y2):
                    reward += zone['reward']
                    self.visited_zones.add(z_id)
                    print(f"\nReached Checkpoint {z_id}!")

        # Environment rewards
        reward += vizdoom_reward

        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_dist = curr_dist
        
        return state, reward, done

    def reset(self):
        self.game.new_episode()
        self.ticks = 0
        self.visited_zones.clear()
        
        self.prev_x = self.game.get_game_variable(GameVariable.POSITION_X)
        self.prev_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.prev_dist = self._calc_dist(self.prev_x, self.prev_y)
        
        return self.preprocess(self.game.get_state().screen_buffer)

    def preprocess(self, img):
        img = cv2.resize(img, (42, 42))
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, :, :]