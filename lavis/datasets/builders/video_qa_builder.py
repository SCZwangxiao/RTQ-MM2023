"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset, NeXtQADataset


class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset


@registry.register_builder("msrvtt_qa")
class MSRVTTQABuilder(VideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa.yaml",
    }


@registry.register_builder("msvd_qa")
class MSVDQABuilder(VideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa.yaml",
    }


@registry.register_builder("nextqa")
class NeXtQABuilder(BaseDatasetBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nextqa/defaults.yaml",
    }
    train_dataset_cls = NeXtQADataset
    eval_dataset_cls = NeXtQADataset

    def build(self):
        datasets = super().build()

        return datasets