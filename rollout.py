import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from A3C import A3C
from Env import create_doom_env


MODEL_PATH = "checkpoints/doom_a3c_ep5000.pth"
# MODEL_PATH = "best/doom_a3c.pth" 
CONFIG_PATH = "speedrun.cfg"
WAD_PATH = "doom.wad"
EPISODES = 5
FRAME_REPEAT = 4
STACK_SIZE = 4

def preprocess(img):
    img = cv2.resize(img, (42, 42))
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, :, :]

def get_action(model, state, game_vars, hidden_state, deterministic=False):
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    vars_tensor = torch.from_numpy(game_vars).float().unsqueeze(0)
    
    with torch.no_grad():
        _, logit, (hx, cx) = model(state_tensor, vars_tensor, hidden_state)
    
    probs = F.softmax(logit, dim=1)
    
    if deterministic:
        action_idx = probs.argmax(dim=1).item()
    else:
        action_idx = probs.multinomial(num_samples=1).item()
        
    actions = ["Left", "Right", "Shoot", "Forward", "Turn Left", "Turn Right", "Use"]
    #print(f"Chosen Action: {actions[action_idx]} | Probabilities: {probs.squeeze().numpy()}")

    return action_idx, (hx, cx)

def main():
    print(f"Loading model from: {MODEL_PATH}")
    # create the environment with stacked frames
    env = create_doom_env(frame_skip=FRAME_REPEAT, visible=True)

    model = A3C(num_inputs=STACK_SIZE, action_space=7, input_size=42)
    
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

    actions = np.eye(7, dtype=np.uint8).tolist()

    for ep in range(EPISODES):
        state, game_vars = env.reset()
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        step = 0
        total_reward = 0
        done = False

        while not done:
            action_idx, (hx, cx) = get_action(model, state, game_vars, (hx, cx))
            state, game_vars, reward, _, done = env.step(action_idx)
            total_reward += reward
            time.sleep(0.02)
            step += 1

        print(f"Episode {ep+1} Finished. Total Reward: {total_reward}")
        time.sleep(1)

    try:
        if hasattr(env, "game"):
            env.game.close()
        else:
            env.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()