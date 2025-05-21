import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm

from constants import Constants


def slice_signal(file, segment_frames, hop_frames):
    mix, _ = librosa.load(file, sr=Constants.SAMPLE_RATE, mono=True)

    slices = []
    mix_frames = len(mix)
    for end_idx in range(segment_frames, mix_frames, hop_frames):
        slice_sig = mix[end_idx - segment_frames:end_idx]
        slices.append(slice_sig)

    return slices, mix_frames


def process_and_serialize(source_root: str,
                          target_root: str,
                          segment_frames: int,
                          hop_frames: int):
    if not os.path.exists(target_root):
        os.mkdir(target_root)
        print(f'Created directory {target_root}')

    for dataset_split in ['train', 'val']:
        target_ds_path = os.path.join(target_root, dataset_split)
        if not os.path.exists(target_ds_path):
            os.mkdir(target_ds_path)
            print(f'Created directory {target_ds_path}')

        source_ds_path = os.path.join(source_root, dataset_split)
        for audiofile in tqdm(os.listdir(os.path.join(source_ds_path, 'clean'))):
            if not audiofile.endswith('.wav'):
                continue

            clean_audiofile_full_path = os.path.join(source_ds_path, 'clean', audiofile)
            noisy_audiofile_full_path = os.path.join(source_ds_path, 'noisy', audiofile)
            if not os.path.exists(noisy_audiofile_full_path):
                continue

            clean_sliced, clean_frames = slice_signal(clean_audiofile_full_path, segment_frames, hop_frames)
            noisy_sliced, noisy_frames = slice_signal(noisy_audiofile_full_path, segment_frames, hop_frames)
            if not clean_frames == noisy_frames:
                continue

            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                target_file_path = os.path.join(target_ds_path, f'{audiofile[:-4]}_{idx}.npy')
                np.save(target_file_path, arr=pair)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help="Path to source wav dataset")
    parser.add_argument('--dst', type=str, required=True, help="Path to save npy dataset")
    parser.add_argument('--segment_frames', type=int, required=True, help="Segment frames")
    parser.add_argument('--hop_frames', type=int, required=True, help="Hop frames")
    args = parser.parse_args()

    process_and_serialize(source_root=args.src,
                          target_root=args.dst,
                          segment_frames=args.segment_frames,
                          hop_frames=args.hop_frames)


if __name__ == '__main__':
    main()
