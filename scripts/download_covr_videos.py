# Code based from from https://github.com/lucas-ventura/CoVR/blob/master/tools/scripts/download_covr.py

import json
import numpy as np
import argparse
import requests
import concurrent.futures
from pathlib import Path
from tqdm.auto import tqdm

def request_save(url, save_fp):
    response = requests.get(url, timeout=5)
    if response.status_code == 404:
        print(f"404 Not Found: {url}")
        return
    with open(save_fp, 'wb') as handler:
        handler.write(response.content)

def main(args):
    video_dir = Path(args.data_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json_path, 'r') as f:
        path2url = json.load(f)
    paths = set(path2url.keys())

    # Remove paths that have already been downloaded
    found_paths = list(video_dir.glob('*/*.mp4'))
    found_paths = {str(p.relative_to(video_dir)) for p in found_paths}
    paths = list(paths - found_paths)
    paths.sort()

    # Split paths into partitions
    paths = np.array_split(paths, args.partitions)[args.part]

    for path in paths:
        vid_path = video_dir / path
        vid_dir = vid_path.parent
        vid_dir.mkdir(exist_ok=True)

    path2url = {path: path2url[path] for path in paths}

    # split into batches of 1000
    for i in tqdm(range(0, len(path2url), 1000)):
        path2url_batch = {path: path2url[path] for path in list(path2url.keys())[i:i+1000]}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
            {executor.submit(request_save, url, video_dir / path) for path, url in path2url_batch.items()}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader')
    parser.add_argument('--partitions', type=int, default=1,
                        help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
    parser.add_argument('--part', type=int, default=0,
                        help='Partition number to download where 0 <= part < partitions')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--json_path', type=str,
                        default="covr/data/webvid2m-covr_paths-cvprw_train.json",
                        help='Path to the local JSON file mapping video paths to URLs.')
    parser.add_argument('--processes', type=int, default=32)
    args = parser.parse_args()

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    main(args)