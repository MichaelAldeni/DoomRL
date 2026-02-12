import os
import torch
import torch.multiprocessing as mp
from Worker import Worker
from A3C import A3C
from Optim import SharedAdam
from tqdm import tqdm
from ICM import ICM
import argparse
import re

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='A3C Doom Training')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint file to resume training (e.g., checkpoints/doom_a3c_ep5000.pth)')
    parser.add_argument('--icm-checkpoint', type=str, default=None,
                        help='Path to ICM checkpoint file (optional, will auto-detect if not provided)')
    args = parser.parse_args()
    
    # Fix for Windows specific multiprocessing issues
    mp.set_start_method('spawn', force=True) 
    os.environ["OMP_NUM_THREADS"] = "1"
    
    lr = 0.00005  # Reduced from 0.0001 to prevent policy collapse
    # Use 4-frame stacking (channels) and 84x84 spatial size to match rollout settings
    input_shape = 4
    input_size = 42
    num_actions = 7

    global_model = A3C(input_shape, num_actions, input_size=input_size)
    global_model.share_memory()

    # ICM should use the same input channels; set its feature_size to match the model
    feature_size = global_model.feature_size
    global_icm = ICM(input_shape, num_actions, feature_size=feature_size)
    global_icm.share_memory()

    optimizer = SharedAdam(global_model.parameters(), lr=lr)
    icm_optimizer = SharedAdam(global_icm.parameters(), lr=lr)

    global_ep_counter = mp.Value('i', 0)
    global_max_reward = mp.Value('d', float('-inf'))  # Shared max reward across all workers
    lock = mp.Lock()

    # Load checkpoint if provided
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            global_model.load_state_dict(checkpoint)
            
            # Try to extract episode number from filename (e.g., doom_a3c_ep1000.pth -> 1000)
            match = re.search(r'ep(\d+)', args.checkpoint)
            if match:
                start_episode = int(match.group(1))
                global_ep_counter.value = start_episode
                print(f"Resuming from episode {start_episode}")
            
            # Try to load ICM checkpoint
            icm_checkpoint_path = args.icm_checkpoint
            if not icm_checkpoint_path:
                # Auto-detect ICM checkpoint path
                icm_checkpoint_path = args.checkpoint.replace('doom_a3c', 'doom_icm')
            
            if os.path.exists(icm_checkpoint_path):
                print(f"Loading ICM checkpoint from {icm_checkpoint_path}")
                icm_checkpoint = torch.load(icm_checkpoint_path, map_location='cpu')
                global_icm.load_state_dict(icm_checkpoint)
            else:
                print(f"Warning: ICM checkpoint not found at {icm_checkpoint_path}, starting ICM from scratch")
        else:
            print(f"Warning: Checkpoint file {args.checkpoint} not found, starting from scratch")
    else:
        print("Starting training from scratch")

    processes = []
    num_processes = os.cpu_count()
    FRAME_SKIP = 4
    print(f"Starting training with {num_processes} workers...")
    for rank in range(num_processes):
        p = Worker(global_model, optimizer, global_icm, icm_optimizer, 
                   global_ep_counter, global_max_reward, lock, str(rank), frame_skip=FRAME_SKIP,
                   input_shape=input_shape, input_size=input_size)
        p.start()
        processes.append(p)

    # Progress bar to track episodes completed
    #pbar = tqdm(total=5000)
    #prev = 0
    #try:
    #    while prev < 5000:
    #        with lock:
    #            cur = global_ep_counter.value
    #        if cur > prev:
    #            pbar.update(cur - prev)
    #            prev = cur
    #finally:
    #    pbar.close()

    for p in processes:
        p.join()
