"""
Datasets: AudioDataset (train/val), AudioTestDataset (test).
"""
import os
import numpy as np
import torch
import torchaudio.transforms as tat
from tqdm.auto import tqdm


def get_dataset_config():
    """Lazy import config to avoid circular dependency."""
    import scripts.config as cfg
    return cfg.get_config()


class AudioDataset(torch.utils.data.Dataset):
    """Load train/val MFCC data with transcripts."""

    def __init__(self, root, phonemes=None, context=0, partition="train-clean-100", config=None):
        if phonemes is None:
            from scripts.config import PHONEMES
            phonemes = PHONEMES
        config = config or get_dataset_config()
        self.context = context
        self.phonemes = phonemes
        self.subset = config["subset"]

        self.freq_masking = tat.FrequencyMasking(freq_mask_param=config["freq_mask_param"])
        self.time_masking = tat.TimeMasking(config["time_mask_param"])

        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")

        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))

        subset_size = int(self.subset * len(mfcc_names))
        mfcc_names = mfcc_names[:subset_size]
        transcript_names = transcript_names[:subset_size]

        assert len(mfcc_names) == len(transcript_names)

        self.mfccs, self.transcripts = [], []

        for i in tqdm(range(len(mfcc_names)), desc=f"Load {partition}"):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0, keepdims=True)) / np.std(mfcc, axis=0, keepdims=True)
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]))
            transcript = transcript[1:-1]  # strip [SOS] and [EOS]
            transcript_indices = [self.phonemes.index(p) for p in transcript]
            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)

            self.mfccs.append(mfccs_normalized)
            self.transcripts.append(transcript_indices)

        self.mfccs = torch.cat(self.mfccs)
        self.transcripts = torch.cat(self.transcripts)
        self.length = len(self.mfccs)

        self.mfccs = torch.nn.functional.pad(
            self.mfccs, pad=(0, 0, self.context, self.context), mode="constant", value=0
        )

        print(f"{partition} MFCC shape: {self.mfccs.shape}")
        print(f"{partition} Transcripts shape: {self.transcripts.shape}")

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        if np.random.rand() < 0.70:
            x = x.transpose(1, 2)
            x = self.freq_masking(x)
            x = self.time_masking(x)
            x = x.transpose(1, 2)
        return x, torch.tensor(y)

    def __getitem__(self, ind):
        frames = self.mfccs[ind : ind + (2 * self.context) + 1]
        phonemes = self.transcripts[ind]
        return frames, phonemes


class AudioTestDataset(torch.utils.data.Dataset):
    """Load test MFCC (no transcripts); order is fixed, do not shuffle."""

    def __init__(self, root, phonemes=None, context=0, partition="test-clean"):
        if phonemes is None:
            from scripts.config import PHONEMES
            phonemes = PHONEMES
        self.context = context
        self.phonemes = phonemes

        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.mfccs = []

        for i in tqdm(range(len(mfcc_names)), desc=f"Load {partition}"):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0, keepdims=True)) / np.std(mfcc, axis=0, keepdims=True)
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)
            self.mfccs.append(mfccs_normalized)

        self.mfccs = torch.cat(self.mfccs)
        self.length = len(self.mfccs)
        self.mfccs = torch.nn.functional.pad(
            self.mfccs, pad=(0, 0, self.context, self.context), mode="constant", value=0
        )

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)

    def __getitem__(self, ind):
        frames = self.mfccs[ind : ind + (2 * self.context) + 1]
        return frames
