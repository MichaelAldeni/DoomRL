import os
import time
import torch
import torch.multiprocessing as mp
from Worker import Worker
from A3C import DoomA3C
from Optim import SharedAdam
from Env import create_doom_env, del_doom_env
from tqdm import tqdm
from ICM import ICM

if __name__ == '__main__':
    # Fix for Windows specific multiprocessing issues
    mp.set_start_method('spawn', force=True) 
    os.environ["OMP_NUM_THREADS"] = "1"
    
    lr = 0.0001
    
    # Create a temporary env ONLY to get dimensions, then delete it immediately
    temp_env = create_doom_env()
    input_shape = 1 
    num_actions = 3
    # IMPORTANT: Close the temp env explicitly before spawning workers
    # If using raw DoomGame: temp_env.close() 
    # If using wrapper: temp_env.game.close()
    del_doom_env(temp_env)

    global_model = DoomA3C(input_shape, num_actions)
    global_model.share_memory()

    global_icm = ICM(input_shape, num_actions)
    global_icm.share_memory()

    optimizer = SharedAdam(global_model.parameters(), lr=lr)
    icm_optimizer = SharedAdam(global_icm.parameters(), lr=lr)

    global_ep_counter = mp.Value('i', 0)
    lock = mp.Lock()

    processes = []
    # Reduce number of workers if on Windows to save memory/stability
    num_processes = os.cpu_count()
    # Frames to repeat each action; increasing this reduces env frequency and speeds training
    FRAME_SKIP = 4

    print(f"Starting training with {num_processes} workers...")

    for rank in range(num_processes):
        p = Worker(global_model, optimizer, global_icm, icm_optimizer, 
                   global_ep_counter, lock, str(rank), frame_skip=FRAME_SKIP)

        p.start()
        processes.append(p)

    # Main-process shared progress bar: update from the shared `global_ep_counter`
    pbar = tqdm(total=5000)
    prev = 0
    try:
        while prev < 5000:
            with lock:
                cur = global_ep_counter.value
            if cur > prev:
                pbar.update(cur - prev)
                prev = cur
            #time.sleep(0.1)
    finally:
        pbar.close()

    for p in processes:
        p.join()