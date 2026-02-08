import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from vizdoom import DoomGame, ScreenFormat, ScreenResolution, Mode, GameVariable
from A3C import A3C


MODEL_PATH = "checkpoints/doom_a3c_ep600.pth"
CONFIG_PATH = "speedrun.cfg"
WAD_PATH = "doom.wad"
EPISODES = 5
FRAME_REPEAT = 4

def preprocess(img):
    img = cv2.resize(img, (42, 42))
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, :, :]

def get_action(model, state, hidden_state, deterministic=False):
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    
    with torch.no_grad():
        _, logit, (hx, cx) = model(state_tensor, hidden_state)
    
    probs = F.softmax(logit, dim=1)
    
    if deterministic:
        action_idx = probs.argmax(dim=1).item()
    else:
        action_idx = probs.multinomial(num_samples=1).item()
        
    actions = ["Left", "Right", "Shoot", "Forward", "Backward", "Turn Left", "Turn Right", "Use"]
    print(f"Chosen Action: {actions[action_idx]} | Probabilities: {probs.squeeze().numpy()}")

    return action_idx, (hx, cx)

def main():
    print(f"Loading model from: {MODEL_PATH}")
    
    game = DoomGame()
    game.load_config(CONFIG_PATH)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)    
    game.init()

    model = A3C(num_inputs=1, action_space=8)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        else:
             model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find file {MODEL_PATH}")
        return

    actions = np.eye(8, dtype=np.uint8).tolist()

    for ep in range(EPISODES):
        game.new_episode()
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        step = 0
        total_reward = 0
        
        while not game.is_episode_finished():
            state_raw = game.get_state().screen_buffer
            state = preprocess(state_raw)
            action_idx, (hx, cx) = get_action(model, state, (hx, cx))
            reward = game.make_action(actions[action_idx], FRAME_REPEAT)
            total_reward += reward
            time.sleep(0.02) 
            step += 1

        print(f"Episode {ep+1} Finished. Total Reward: {total_reward}")
        time.sleep(1)

    game.close()

if __name__ == "__main__":
    main()