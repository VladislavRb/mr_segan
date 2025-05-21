import os

import numpy as np
import torch
from torch.utils import data


def emphasis(signal_batch, ec=0.95, pre=True):
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            if pre:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - ec * channel_data[:-1])
            else:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + ec * channel_data[:-1])

    return result


class MRWaveformDataset(data.Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f'The {self.dataset_path} data folder does not exist!')

        self.file_names = [os.path.join(self.dataset_path, filename) for filename in os.listdir(self.dataset_path)]

    def reference_batch(self, batch_size):
        ref_file_names = np.random.choice(self.file_names, batch_size)
        ref_batch = np.stack([np.load(f) for f in ref_file_names])

        ref_batch = emphasis(ref_batch, ec=0.95)
        return torch.from_numpy(ref_batch).type(torch.FloatTensor)

    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx])
        pair = emphasis(pair[np.newaxis, :, :], ec=0.95).reshape(2, -1)

        noisy = pair[1].reshape(1, -1)
        clean = pair[0].reshape(1, -1)

        pair = torch.from_numpy(pair).type(torch.FloatTensor)
        clean = torch.from_numpy(clean).type(torch.FloatTensor)
        noisy = torch.from_numpy(noisy).type(torch.FloatTensor)

        return pair, clean, noisy

    def __len__(self):
        return len(self.file_names)
