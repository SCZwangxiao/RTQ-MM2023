import os
import sys
import csv
import glob
from tqdm import tqdm

import lmdb
import mmengine


def get_video_path():
    vid2path = dict()

    filepaths = glob.glob("./NExTVideo/*/*.mp4")

    for filepath in filepaths:
        vid = filepath.split('/')[-1].split('.')[0]
        assert vid not in vid2path
        vid2path[vid] = filepath

    return vid2path


def build_nextqa_txt_db(data_root, vid2path):
    os.makedirs(os.path.join(data_root, 'txt_db'), exist_ok=True)

    for split in ['train', 'val', 'test']:
        dataset = []

        if split == "test":
            anno_path = "./test-data-nextqa/test.csv"
        else:
            anno_path = f"./nextqa/{split}.csv"

        with open(anno_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # skip the headers
            for row in csv_reader:
                video, frame_count, width, height, question, answer, qid, type, a0, a1, a2, a3, a4 = row
                assert video in vid2path, video
                dataset.append(dict(
                    video=video,
                    qid=qid,
                    type=type,
                    question=question,
                    answer=int(answer),
                    a0=a0,
                    a1=a1,
                    a2=a2,
                    a3=a3,
                    a4=a4,
                ))

        mmengine.dump(dataset, os.path.join(data_root, 'txt_db', f'{split}.json'), indent=2)

    # Merge training set and validation set
    train_data_original = mmengine.load(os.path.join(data_root, 'txt_db', 'train.json'))
    val_data = mmengine.load(os.path.join(data_root, 'txt_db', 'val.json'))
    train_data = train_data_original + val_data
    os.system(f'mv %s %s' % (os.path.join(data_root, 'txt_db', 'train.json'), os.path.join(data_root, 'txt_db', 'train_original.json')))
    mmengine.dump(train_data, os.path.join(data_root, 'txt_db', 'train.json'), indent=2)


def build_nextqa_vis_db(data_root, vid2path):
    vis_db_path = os.path.join(data_root, 'vis_db')
    os.system(f'rm -rf {vis_db_path}')
    os.makedirs(vis_db_path)

    # Open the LMDB database for writing
    env = lmdb.open(vis_db_path, map_size=1099511627776, writemap=True)

    # Start a write transaction
    with env.begin(write=True) as txn:
        # Loop through each video path in the list
        for vid in tqdm(list(vid2path.keys())):
            path = vid2path[vid]
            # Open the video file
            with open(path, 'rb') as F:
                video_bytes = F.read()

            txn.put(vid.encode(), video_bytes)

    # Close the LMDB database
    env.close()


if __name__ == "__main__":
    data_root = sys.argv[1]

    os.makedirs(data_root, exist_ok=True)

    vid2path = get_video_path()

    build_nextqa_txt_db(data_root, vid2path)

    build_nextqa_vis_db(data_root, vid2path)