# This source is based on https://github.com/AbnerHqC/GaitSet/blob/master/pretreatment.py
import argparse
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def imgs2pickle(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format.

    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """    
    sinfo = img_groups[0]
    img_paths = img_groups[1]
    to_pickle = []
    for img_file in sorted(img_paths):
        if verbose:
            logging.debug(f'Reading sid {sinfo[0]}, seq {sinfo[1]}, view {sinfo[2]} from {img_file}')

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        
        if dataset == 'GREW':
            to_pickle.append(img.astype('uint8'))
            continue

        to_pickle.append(img.astype('uint8'))

    if to_pickle:
        to_pickle = np.asarray(to_pickle)
        dst_path = os.path.join(output_path, *sinfo)
        # print(img_paths[0].as_posix().split('/'),img_paths[0].as_posix().split('/')[-5])
        # dst_path = os.path.join(output_path, img_paths[0].as_posix().split('/')[-5], *sinfo) if dataset == 'GREW' else dst
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            logging.debug(f'Saving {pkl_path}...')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))   
        logging.info(f'Saved {len(to_pickle)} valid frames to {pkl_path}.')


    if len(to_pickle) < 5:
        logging.warning(f'{sinfo} has less than 5 valid data.')



def pretreat(input_path: Path, output_path: Path, img_size: int = 64, workers: int = 4, verbose: bool = False, dataset: str = 'CASIAB') -> None:
    """Reads a dataset and saves the data in pickle format.

    Args:
        input_path (Path): Dataset root path.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    img_groups = defaultdict(list)
    logging.info(f'Listing {input_path}')
    total_files = 0
    for img_path in input_path.rglob('*.png'):
        if 'gei.png' in img_path.as_posix():
            continue
        if verbose:
            logging.debug(f'Adding {img_path}')
        *_, sid, seq, view, _ = img_path.as_posix().split('/')
        img_groups[(sid, seq, view)].append(img_path)
        total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(partial(imgs2pickle, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset), img_groups.items()):
            progress.update(1)
    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='', type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=32, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-r', '--img_size', default=64, type=int, help='Image resizing size. Default 64')
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), img_size=args.img_size, workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)
# python datasets/pretreatment_wo_cut.py --input_path /home/lxc/dataset/CASIA_B/silhouettes_seg_cut --output_path /home/lxc/dataset/CASIA_B/silhouettes_seg_cut_pkl