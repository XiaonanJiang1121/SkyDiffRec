import torch
import numpy as np
import pdb
from .utils import dist_util
from torch import nn





class WrappedModel(nn.Module):
    def __init__(self, original_num_steps):
        super().__init__()
        self.original_num_steps = original_num_steps

    def forward(self, batch_size):
        ts, weights = self.sample(batch_size, dist_util.dev())
        #map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        #new_ts = map_tensor[ts]
        # print(new_ts)
        #if self.rescale_timesteps:
        #    new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        # temp = self.model(x, new_ts, **kwargs)
        # print(temp.shape)
        # return temp
        # print(new_ts)
        #pdb.set_trace()
        return ts, weights

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = np.ones(self.original_num_steps)
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


