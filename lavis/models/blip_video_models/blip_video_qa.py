"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lavis.common.utils import is_url
from lavis.common.registry import registry
from lavis.common.dist_utils import download_cached_file
from lavis.models.backbones.vit import interpolate_pos_embed
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.models.blip_models.blip_vqa import BlipVQA
from lavis.models.blip_video_models.blip_video_outputs import BlipVideoIntermediateOutput


@registry.register_model("blip_video_qa")
class BlipVideoQA(BlipVQA):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/blip_post/blip_base_mc_qa.yaml",
    }

    def forward(self, samples):
        encoder_output, video_embeds = self.forward_encoder(samples)
        loss, decoder_output, decoder_targets = self.forward_decoder(
            samples=samples, encoder_out=encoder_output
        )

        return BlipOutput(
            loss=loss,
            intermediate_output=BlipVideoIntermediateOutput(
                video_embeds=video_embeds,
                encoder_output=encoder_output,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )

    def forward_encoder(self, samples):
        questions = samples["text_input"]
        questions = self.tokenizer(
            questions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        samples.update({"tokenized_text": questions})

        video_embeds = self.visual_encoder.forward_features(samples["video"])[1]
        encoder_output = self.text_encoder.forward_automask(
            tokenized_text=samples["tokenized_text"], visual_embeds=video_embeds
        )

        return encoder_output, video_embeds

    def forward_decoder(self, samples, encoder_out, **kwargs):
        answers = self.tokenizer(
            samples["answer"], padding="longest", return_tensors="pt"
        ).to(self.device)
        answers.input_ids[:, 0] = self.tokenizer.bos_token_id
        answer_targets = answers.input_ids.masked_fill(
            answers.input_ids == self.tokenizer.pad_token_id, -100
        )

        question_states = []
        question_atts = []

        question = samples["tokenized_text"]
        question_output = encoder_out

        for b, n in enumerate(samples["n_answers"]):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n

        question_states = torch.stack(question_states, dim=0)
        question_atts = torch.stack(question_atts, dim=0)

        answer_output = self.text_decoder(
            answers.input_ids,
            attention_mask=answers.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=answer_targets,
            return_dict=True,
            reduction="none",
        )

        loss = samples["weight"] * answer_output.loss
        bsz = samples["video"].size(0)

        loss = loss.sum() / bsz

        return loss, answer_output, answer_targets

    def predict_answers(
        self,
        samples,
        num_beams=3,
        inference_method="rank",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        **kwargs
    ):
        assert inference_method in [
            "rank",
            "generate",
        ], "Inference method must be one of 'rank' or 'generate', got {}.".format(
            inference_method
        )

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["video"].size(
            0
        ), "The number of questions must be equal to the batch size."

        if inference_method == "generate":
            return self._generate_answers(
                samples, num_beams=num_beams, max_length=max_len, min_length=min_len
            )
        elif inference_method == "rank":
            assert answer_list is not None, "answer_list must be provided for ranking"

            num_ans_candidates = min(num_ans_candidates, len(answer_list))

            return self._rank_answers(
                samples, answer_list=answer_list, num_ans_candidates=num_ans_candidates
            )

    def _generate_answers(self, samples, num_beams=3, max_length=10, min_length=1):
        encoder_out, _ = self.forward_encoder(samples)

        question_output = encoder_out

        # question_states = question_output.last_hidden_state.repeat_interleave(
        #     num_beams, dim=0
        # )
        question_states = question_output.last_hidden_state
        question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(
            self.device
        )

        model_kwargs = {
            "encoder_hidden_states": question_states,
            "encoder_attention_mask": question_atts,
        }

        bsz = samples["video"].size(0)
        bos_ids = torch.full(
            (bsz, 1), fill_value=self.tokenizer.bos_token_id, device=self.device
        )

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )

        # collect answers
        answers = []
        for output in outputs:
            answer = self.tokenizer.decode(output, skip_special_tokens=True)
            answers.append(answer)

        return answers
    
    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        if hasattr(self.visual_encoder, 'modify_state_dict'):
            logging.info("Modify state dict according to visual encoder {}.".format(type(self.visual_encoder)))
            state_dict = self.visual_encoder.modify_state_dict(state_dict)
        if hasattr(self.text_encoder, 'modify_state_dict'):
            state_dict = self.text_encoder.modify_state_dict(state_dict)
            logging.info("Modify state dict according to text encoder {}.".format(type(self.text_encoder)))

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        if hasattr(self.visual_encoder, 'modify_state_dict'):
            logging.info("Modify state dict according to image encoder {}.".format(type(self.visual_encoder)))
            state_dict = self.visual_encoder.modify_state_dict(state_dict)
        if hasattr(self.text_decoder, 'modify_state_dict'):
            state_dict = self.text_decoder.modify_state_dict(state_dict)
            logging.info("Modify state dict according to text encoder {}.".format(type(self.text_decoder)))

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )

        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg