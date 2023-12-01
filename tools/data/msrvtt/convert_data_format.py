import os
import sys
import json


def load_json(file):
    with open(file, 'r') as F:
        data = json.load(F)
    return data


def write_jsonline(data, file):
    assert type(data) == list
    assert type(data[0]) == dict

    with open(file, "w") as F:
        for item in data:
            json.dump(item, F)
            F.write('\n')


if __name__ == "__main__":
    data_root = sys.argv[1]

    for split in ['train', 'val', 'test']:
        file = os.path.join(data_root, f'cap_{split}.json')
        new_file = os.path.join(data_root, f'{split}.jsonl')
        data = load_json(file)

        new_data = []
        vid2captions = {}
        for d in data:
            vid = d['image_id'].split('.')[0]
            caption = d['caption']
            captions = vid2captions.get(vid, [])
            captions.append(caption)
            vid2captions[vid] = captions
        for vid, captions in vid2captions.items():
            new_data.append(dict(
                clip_name=vid,
                caption=captions
            ))

        write_jsonline(new_data, new_file)

