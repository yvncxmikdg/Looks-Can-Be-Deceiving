import torch

def reseed_distributed(index, random_state, seed_range=10e6):
    worker_info = torch.utils.data.get_worker_info()
    wid = 0 if worker_info is None else worker_info.id
    mod_index = index % seed_range
    seed = int(mod_index + (wid * seed_range))
    random_state.seed(seed)
