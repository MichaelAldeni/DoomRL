from vizdoom import DoomGame, GameVariable
import numpy as np
import cv2

def create_doom_env(frame_skip: int = 4):
    """
    Creates and configures the VizDoom environment.

    Args:
        frame_skip (int): number of frames to repeat each chosen action (frame-skip).
    """
    game = DoomGame()
    game.load_config("speedrun.cfg")
    game.init()
    return DoomEnvWrapper(game, frame_skip)

def del_doom_env(env):
    # Close the underlying DoomGame instance correctly
    try:
        if hasattr(env, "game") and env.game is not None:
            env.game.close()
        else:
            env.close()
    except Exception:
        pass
    return

class DoomEnvWrapper:
    """Wrapper to handle resizing and logic"""
    def __init__(self, game, frame_skip: int = 4):
        self.game = game
        self.frame_skip = int(frame_skip)
        # Left, Right, Shoot
        self.actions = np.eye(8, dtype=np.uint8).tolist()

        # E1M1 Exit Switch Coordinates (approximate)
        self.GOAL_X = 2955.0
        self.GOAL_Y = -4772.0
        
        # Exploration tracking
        # Format: (Min_X, Min_Y, Max_X, Max_Y, Reward_Value)
        # Coordinates below are examples for E1M1. You can find yours by printing curr_x/y.
        self.checkpoints = [
            # Example: The hallway after the first door in E1M1
            {'id': 1, 'coords': (1568, -2560, 1664, -2432), 'reward': 50.0},
            # Example: The Zig-Zag room
            {'id': 2, 'coords': (2944, -4160, 3072, -4032), 'reward': 50.0}
        ]
        self.visited_zones = set()
        
        # State tracking for reward calculation
        self.prev_x = 0
        self.prev_y = 0
        self.prev_dist = 0
        
    def _calc_dist(self, x, y):
        return np.sqrt((x - self.GOAL_X)**2 + (y - self.GOAL_Y)**2)

    def step(self, action_idx):
        # 1. Execute Action
        # Repeat the chosen action for `self.frame_skip` frames to speed up training
        # Capture the reward from VizDoom (includes living_reward and death_penalty from config)
        vizdoom_reward = self.game.make_action(self.actions[action_idx], self.frame_skip)
        
        # 2. Get Game State
        done = self.game.is_episode_finished()
        if done:
            # If done, we can't get variables, so we return dummy state and finish reward
            state = np.zeros((1, 42, 42), dtype=np.float32)
            # Big bonus for actually finishing the level (hitting the switch)
            # We assume finishing means > 1 minute survival or specific game state, 
            # but usually 'done' in Doom means death or exit. 
            # If you want to differentiate death vs win, check game.get_game_variable(GameVariable.HITCOUNT) or similar,
            # or just rely on the fact that reaching the goal yields high reward naturally.
            return state, 100.0, done
            
        state = self.preprocess(self.game.get_state().screen_buffer)
        
        # 3. Calculate Reward
        curr_x = self.game.get_game_variable(GameVariable.POSITION_X)
        curr_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        curr_dist = self._calc_dist(curr_x, curr_y)
        
        reward = 0.0
        
        # A. Goal Homing (Shaping Reward)
        # Reward = Improvement in distance. 
        # If we got closer: positive. If we moved away: negative.
        dist_improvement = (self.prev_dist - curr_dist)
        reward += dist_improvement * 1.0 
        
        # B. Velocity (Encourage Movement)
        # Calculate distance traveled this step
        travel_dist = np.sqrt((curr_x - self.prev_x)**2 + (curr_y - self.prev_y)**2)
        if travel_dist > 0.5: # Filter jitter
             reward += travel_dist * 0.01
        
        # Penalty for turning (actions 5, 6) without moving - prevents ICM exploiting visual novelty from spinning
        if action_idx in (2, 5, 6, 7) and travel_dist < 1.0:
            reward -= 0.1
        
        # C. Exploration (Checkpoints)
        # --- A. SECTION REWARDS (One-time Bonus) ---
        for zone in self.checkpoints:
            z_id = zone['id']
            if z_id not in self.visited_zones:
                x1, y1, x2, y2 = zone['coords']
                # Check if inside the bounding box (use min/max to handle any coordinate order)
                if min(x1, x2) <= curr_x <= max(x1, x2) and min(y1, y2) <= curr_y <= max(y1, y2):
                    reward += zone['reward']
                    self.visited_zones.add(z_id)
                    print(f"Reached Checkpoint {z_id}!")

        # D. Add VizDoom's native rewards (living_reward, death_penalty, etc.)
        reward += vizdoom_reward

        # Update previous values for next step
        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_dist = curr_dist
        
        # Normalize reward usually helps (clip it)
        # reward = max(min(reward, 1.0), -1.0) 
        
        return state, reward, done

    def reset(self):
        self.game.new_episode()
        self.visited_zones.clear()  # Reset checkpoint tracking each episode
        
        # Initialize coordinates
        self.prev_x = self.game.get_game_variable(GameVariable.POSITION_X)
        self.prev_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.prev_dist = self._calc_dist(self.prev_x, self.prev_y)
        
        return self.preprocess(self.game.get_state().screen_buffer)

    def preprocess(self, img):
        img = cv2.resize(img, (42, 42))
        img = img.astype(np.float32) / 255.0
        # Add channel dimension: (42, 42) -> (1, 42, 42)
        return img[np.newaxis, :, :]