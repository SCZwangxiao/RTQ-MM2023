"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import io
from mmengine.fileio import FileClient 

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset


def reforge_annotations(old_annotation):
    annotation = []
    for anno in old_annotation:
        # Get video id
        if 'clip_name' in anno: # datasets from clipbert
            video_id = anno['clip_name']
        else:
            raise NotImplementedError

        captions = anno.pop('caption')
        new_anno = dict(video=video_id, caption=captions)
        new_anno.update(anno)
        annotation.append(new_anno)

    return annotation


def explode_annotations(old_annotation):
    annotation = []
    for anno in old_annotation:
        # explode single video - multi captions
        if type(anno['caption']) != list:
            anno['caption'] = [anno['caption']]
        captions = anno.pop('caption')
        for caption in captions:
            new_anno = dict(caption=caption)
            new_anno.update(anno)
            annotation.append(new_anno)

    return annotation


class VideoCaptionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Forge annotations
        self.annotation = reforge_annotations(self.annotation)

        # Explode annotations for training
        self.annotation = explode_annotations(self.annotation)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["video"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        # Get file client
        self.file_client = FileClient(backend='lmdb', db_path=vis_root)

    def __getitem__(self, index):
        ann = self.annotation[index]

        vid_obj = io.BytesIO(self.file_client.get(ann["video"]))

        video = self.vis_processor(vid_obj)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["video"]],
        }


class VideoCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Forge annotations
        self.annotation = reforge_annotations(self.annotation)

        # Get file client
        self.file_client = FileClient(backend='lmdb', db_path=vis_root)

    def __getitem__(self, index):
        ann = self.annotation[index]

        vid_obj = io.BytesIO(self.file_client.get(ann["video"]))

        video = self.vis_processor(vid_obj)

        return {
            "video": video,
            "image_id": ann["video"],
            "instance_id": ann["instance_id"],
        }
