"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import os
from collections import OrderedDict

from PIL import Image
from mmengine.fileio import FileClient 

from lavis.datasets.datasets.base_dataset import BaseDataset


def reforge_annotations(old_annotation):
    annotation = []
    for anno in old_annotation:
        # Get video id
        if 'clip_name' in anno: # datasets from clipbert
            video_id = anno['clip_name']
        elif 'video' in anno: # nextqa dataset
            video_id = anno['video']
        else:
            raise NotImplementedError
        # Dealwith multi-caption per video (MSRVTT)
        if type(anno['caption']) != list:
            anno['caption'] = [anno['caption']]
        captions = anno.pop('caption')
        for caption in captions:
            new_anno = dict(video=video_id, caption=caption)
            new_anno.update(anno)
            annotation.append(new_anno)

    return annotation


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class RetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
        }


class RetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}


class VideoRetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Forge annotations
        self.annotation = reforge_annotations(self.annotation)

        # Get file client
        self.file_client = FileClient(backend='lmdb', db_path=vis_root)

        # Generate image ids
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["video"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vid_obj = io.BytesIO(self.file_client.get(ann["video"]))

        video = self.vis_processor(vid_obj)
        caption = self.text_processor(ann["caption"])

        return {
            "video": video,
            "text_input": caption,
            "video_id": self.img_ids[ann["video"]],
        }


class VideoRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Forge annotations
        self.annotation = reforge_annotations(self.annotation)

        # Get file client
        self.file_client = FileClient(backend='lmdb', db_path=vis_root)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["video"])
            self.text.append(self.text_processor(ann["caption"]))
            self.img2txt[img_id] = [txt_id]
            self.txt2img[txt_id] = img_id
            txt_id += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        vid_obj = io.BytesIO(self.file_client.get(ann["video"]))

        video = self.vis_processor(vid_obj)

        return {"video": video, "index": index}
