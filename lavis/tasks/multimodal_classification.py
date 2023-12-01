"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import logging

import numpy as np
import pandas as pd
import torch
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


def calculate_type_accuracy(issame, types):
    df = pd.DataFrame({
        'issame': issame,
        'types': types
    })
    df = df.groupby(['types']).mean()
    type2acc = df.to_dict()['issame']
    return type2acc


def nextqa_accuracy(preds, 
                    labels, 
                    types,
                    root_category=True,
                    second_category=True,
                    third_category=False):
    """
        preds: [b]
        labels: [b]
        types: [b]

        '' stands for all types
    """
    type2acc = {}
    issame = (preds == labels)
    issame = issame.astype(np.int32)

    if third_category:
        res = calculate_type_accuracy(issame, types)
        type2acc.update(res)
    
    if second_category:
        types = [t[0] for t in types]
        res = calculate_type_accuracy(issame, types)
        type2acc.update(res)
    
    if root_category:
        types = ['all' for t in types]
        res = calculate_type_accuracy(issame, types)
        type2acc.update(res)
    
    return type2acc


@registry.register_task("multimodal_classification")
class MultimodalClassificationTask(BaseTask):
    def __init__(self):
        super().__init__()

    def valid_step(self, model, samples):
        results = []

        outputs = model.predict(samples)

        predictions = outputs["predictions"]
        targets = outputs["targets"]
        question_types = samples.get("question_type", ['all'] * predictions.shape[0])

        predictions = predictions.max(1)[1].cpu().numpy()
        targets = targets.cpu().numpy()

        indices = samples[self.inst_id_key]

        for pred, tgt, index, qtype in zip(predictions, targets, indices, question_types):
            if isinstance(index, torch.Tensor):
                index = index.item()

            results.append(
                {
                    self.inst_id_key: index,
                    "prediction": pred.item(),
                    "target": tgt.item(),
                    "qtype": qtype,
                }
            )

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=self.inst_id_key,
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        results = json.load(open(eval_result_file))

        predictions = np.array([res["prediction"] for res in results])
        targets = np.array([res["target"] for res in results])
        types = [res["qtype"] for res in results]

        type2acc = nextqa_accuracy(predictions, targets, types)
        metrics = {}
        for type, acc in type2acc.items():
            acc *= 100
            if type == 'all':
                metrics[f'acc'] = acc
                metrics['agg_metrics'] = acc
            else:
                metrics[f'acc-{type}'] = acc

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
