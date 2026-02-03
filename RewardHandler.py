import numpy as np
from vizdoom import GameVariable

class SpeedrunRewardHandler:
    def __init__(self, game):
        self.game = game
        
        # E1M1 Exit Switch approximate coordinates 
        # (You may need to verify these exact coords in a map editor like SLADE)
        self.GOAL_X = 2955.0
        self.GOAL_Y = -4772.0
        
        # State tracking
        self.prev_x = self.game.get_game_variable(GameVariable.POSITION_X)
        self.prev_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.prev_dist_to_goal = self._calc_dist(self.prev_x, self.prev_y)
        
        # Track visited sectors (areas) to encourage exploration
        # requires GameVariable.POSITION_X/Y/Z to be reliable
        self.visited_sectors = set() 
        
    def _calc_dist(self, x, y):
        """Calculates Euclidean distance to the specific goal coordinate."""
        return np.sqrt((x - self.GOAL_X)**2 + (y - self.GOAL_Y)**2)

    def calculate_reward(self):
        """
        Call this function immediately AFTER making an action (game.make_action).
        """
        # 1. Get current state
        curr_x = self.game.get_game_variable(GameVariable.POSITION_X)
        curr_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        # 2. Calculate basic movement (Velocity)
        dist_covered_step = np.sqrt((curr_x - self.prev_x)**2 + (curr_y - self.prev_y)**2)
        
        # 3. Calculate progress toward the specific goal
        curr_dist_to_goal = self._calc_dist(curr_x, curr_y)
        dist_improvement = self.prev_dist_to_goal - curr_dist_to_goal
        
        # Initialize Reward
        reward = 0.0
        
        # --- REWARD COMPONENT A: Velocity (Momentum) ---
        # Reward simply moving fast (teaches strafe running)
        # We clamp this to avoid rewarding teleportation or glitches
        if dist_covered_step < 50.0: 
            reward += dist_covered_step * 0.01

        # --- REWARD COMPONENT B: Goal Homing (Dense Reward) ---
        # If we got closer to the exit than we were a moment ago, give a bonus.
        # If we moved away, give a penalty.
        reward += dist_improvement * 0.5
        
        # --- REWARD COMPONENT C: Exploration (Breadcrumbs) ---
        # We use integer coordinates as a rough "grid" to identify new areas
        current_sector_hash = (int(curr_x // 100), int(curr_y // 100))
        if current_sector_hash not in self.visited_sectors:
            reward += 10.0  # Big bonus for finding a new room
            self.visited_sectors.add(current_sector_hash)

        # --- REWARD COMPONENT D: Time Pressure ---
        # Penalize existing. The longer it takes, the lower the score.
        reward -= 0.1 

        # --- REWARD COMPONENT E: The Finish Line ---
        if self.game.is_episode_finished():
            # Check if we actually finished (hit the switch) vs died
            # Usually implies checking game variables or simply if reward > threshold
            # But simpler: A massive bonus for the episode ending naturally
            reward += 1000.0

        # Update state for next step
        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_dist_to_goal = curr_dist_to_goal
        
        return reward