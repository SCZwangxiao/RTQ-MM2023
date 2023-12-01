"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import io
import json
from collections import OrderedDict

import torch
from mmengine.fileio import FileClient

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from lavis.datasets.datasets.vqa_datasets import VQADataset


class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Get file client
        self.file_client = FileClient(backend='lmdb', db_path=vis_root)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vid_id = ann["video"]
        if vid_id.endswith('.mp4'):
            vid_id = vid_id[:-4]
        vid_obj = io.BytesIO(self.file_client.get(vid_id))
        video = self.vis_processor(vid_obj)

        question = self.text_processor(ann["question"])

        # assert n_answers == 1
        assert type(ann["answer"]) == str

        return {
            "video": video,
            "text_input": question,
            "answers": ann["answer"],
            "weights": 1.,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []
        num_answers = []
        question_id_list = [int(sample["question_id"]) for sample in samples]

        for sample in samples:
            image_list.append(sample["video"])
            question_list.append(sample["text_input"])

            weight_list.append(sample["weights"])

            answers = sample["answers"]

            answer_list.append(answers)
            num_answers.append(1)

        return {
            "video": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
            "question_id": torch.LongTensor(question_id_list),
        }


class NeXtQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Get file client
        self.file_client = FileClient(backend='lmdb', db_path=vis_root)

    def __getitem__(self, index):

        anno = self.annotation[index]

        vid_obj = io.BytesIO(self.file_client.get(anno["video"]))
        video = self.vis_processor(vid_obj)

        question = anno["question"]

        text_input = [f'question: {question}? answer: {anno[a]}' for a in ['a0', 'a1', 'a2', 'a3', 'a4']]

        question_type = anno['type']

        text_input = [self.text_processor(text) for text in text_input]

        return {
            "video": video,
            "text_input": text_input,
            "label": anno['answer'],
            "question_type": question_type,
            "question_id": anno["qid"],
            "instance_id": anno["instance_id"],
        }
