import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from vizdoom import DoomGame, ScreenFormat, ScreenResolution, Mode
from A3C import DoomA3C

# --- Configuration ---
MODEL_PATH = "checkpoints/doom_a3c_ep1000.pth"  # <--- CHANGE THIS to your actual file
CONFIG_PATH = "speedrun.cfg"           # The config you created earlier
WAD_PATH = "DOOM.wad"                       # Standard VizDoom wad
EPISODES_TO_WATCH = 5
FRAME_REPEAT = 4

def preprocess(img):
    """
    Must match the training preprocessing EXACTLY.
    Resize to 42x42, Grayscale, Normalize 0-1.
    """
    img = cv2.resize(img, (42, 42))
    img = img.astype(np.float32) / 255.0
    return img

def get_action(model, state, hidden_state, deterministic=True):
    """
    Pass state through model and decide action.
    """
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        _, logit, (hx, cx) = model(state_tensor, hidden_state)
    
    probs = F.softmax(logit, dim=1)
    
    if deterministic:
        # Best action (Greedy)
        action_idx = probs.argmax(dim=1).item()
    else:
        # Random action based on probability (Stochastic)
        action_idx = probs.multinomial(num_samples=1).item()
        
    return action_idx, (hx, cx)

def main():
    print(f"Loading model from: {MODEL_PATH}")
    
    # 1. Initialize Doom Game for Testing
    game = DoomGame()
    game.load_config(CONFIG_PATH)
    game.set_doom_scenario_path(WAD_PATH)
    
    # Visuals: We want to see the game now!
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER) # Change to ASYNC_PLAYER if you want to play yourself
    
    # Screen Format overrides (just in case cfg is different)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480) 
    
    game.init()

    # 2. Load Model
    # 1 input channel (Grayscale), 3 actions (Left, Right, Shoot)
    model = DoomA3C(num_inputs=1, action_space=3)
    
    # Load weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # Handle case where checkpoint was saved as full dict or just state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        else:
             model.load_state_dict(checkpoint)
        model.eval() # Set to evaluation mode (disable dropout, etc)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find file {MODEL_PATH}")
        return

    # Action definition (Identity matrix for One-Hot)
    actions = np.eye(3, dtype=np.uint8).tolist()

    # 3. Game Loop
    for ep in range(EPISODES_TO_WATCH):
        game.new_episode()
        
        # Reset LSTM hidden state at start of episode
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        step = 0
        total_reward = 0
        
        while not game.is_episode_finished():
            # Get State
            state_raw = game.get_state().screen_buffer
            state = preprocess(state_raw)
            
            # Get Action
            action_idx, (hx, cx) = get_action(model, state, (hx, cx))
            
            # Make Action
            reward = game.make_action(actions[action_idx], FRAME_REPEAT)
            total_reward += reward
            
            # Optional: Sleep to slow down playback if it's too fast
            time.sleep(0.02) 
            step += 1

        print(f"Episode {ep+1} Finished. Total Reward: {total_reward}")
        time.sleep(1) # Pause between episodes

    game.close()

if __name__ == "__main__":
    main()