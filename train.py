import os
import torch.multiprocessing as mp
from Worker import Worker
from A3C import A3C
from Optim import SharedAdam
from tqdm import tqdm
from ICM import ICM

if __name__ == '__main__':
    # Fix for Windows specific multiprocessing issues
    mp.set_start_method('spawn', force=True) 
    os.environ["OMP_NUM_THREADS"] = "1"
    
    lr = 0.0001
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