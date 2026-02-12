import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
from A3C import A3C
from Env import create_doom_env


# MODEL_PATH = "checkpoints/doom_a3c_ep5000.pth"
MODEL_PATH = "best/doom_a3c_784.79.pth" 
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
    env = create_doom_env(frame_skip=FRAME_REPEAT, visible=False)

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

    episode_num = 0
    victory_video = None
    
    while True:
        # Start recording the episode
        episode_num += 1
        video_path = f"episode_{episode_num}.avi"
        video_writer = None
        
        state, game_vars = env.reset()
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        step = 0
        total_reward = 0
        done = False

        while not done:
            # Get the current frame for recording
            if env.game.get_state() is not None:
                frame = env.game.get_state().screen_buffer
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Initialize video writer on first frame
                if video_writer is None:
                    height, width = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                
                video_writer.write(frame_bgr)
            
            action_idx, (hx, cx) = get_action(model, state, game_vars, (hx, cx))
            state, game_vars, reward, _, done = env.step(action_idx)
            total_reward += reward
            time.sleep(0.02)
            step += 1

        # Release video writer
        if video_writer is not None:
            video_writer.release()
        
        # Check if this was a victory (total_reward >= 700 indicates success)
        is_victory = total_reward >= 700
        
        if is_victory:
            print(f"Episode {episode_num} - VICTORY! Total Reward: {total_reward}")
            # Delete previous victory video if exists
            if victory_video and os.path.exists(victory_video):
                os.remove(victory_video)
                print(f"Deleted previous victory video: {victory_video}")
            
            # Rename current video to victory video
            victory_video = f"victory_episode_{episode_num}.avi"
            os.rename(video_path, victory_video)
            print(f"Saved victory video: {victory_video}")
            
            # Delete all non-victory episode videos
            for vid_file in glob.glob("episode_*.avi"):
                if os.path.exists(vid_file):
                    os.remove(vid_file)
                    print(f"Deleted: {vid_file}")
        else:
            print(f"Episode {episode_num} - Failed. Total Reward: {total_reward}")
            # Delete the failed episode video
            if os.path.exists(video_path):
                os.remove(video_path)
        
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