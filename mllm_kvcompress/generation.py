# SPDX-License-Identifier: Apache-2.0
"""Question answering on a compressed multimodal context, as a transformers pipeline."""

import logging
from typing import Optional

import torch
from transformers import Cache, DynamicCache, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY

from mllm_kvcompress.core.cache import (
    kv_cache_layer_indices,
    language_model_of,
    multimodal_backbone_of,
    per_layer_attention_masks,
    restore_cache,
    snapshot_cache,
)
from mllm_kvcompress.core.runtime import CompressionMethod, compress

logger = logging.getLogger(__name__)

PIPELINE_NAME = "multimodal-kv-compression"


def new_dynamic_cache(model) -> DynamicCache:
    """Create a DynamicCache with model config when required by hybrid caches."""
    try:
        return DynamicCache(config=language_model_of(model).config)
    except TypeError:
        return DynamicCache()


class CompressedContextPipeline(Pipeline):
    """
    Pipeline answering questions about a compressed multimodal context.

    The multimodal context (images + optional context text) is pre-filled with KV
    cache compression applied, then one or several questions are answered with greedy
    decoding on the compressed cache. The question is excluded from compression, so
    methods are evaluated on a cache that must work for any follow-up query.

    Example
    -------
    >>> from transformers import pipeline
    >>> pipe = pipeline("multimodal-kv-compression", model=model_name, device="cuda")
    >>> result = pipe(image, question="What is in the image?", method=LookM(ratio=0.5))
    >>> result["answer"]
    """

    _load_processor = True
    _load_tokenizer = True
    _load_image_processor = False
    _load_feature_extractor = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.processor is not None, "CompressedContextPipeline requires a processor"
        if self.tokenizer is None:
            self.tokenizer = self.processor.tokenizer

    def _sanitize_parameters(
        self,
        context: Optional[str] = None,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: str = "",
        method: Optional[CompressionMethod] = None,
        max_new_tokens: int = 100,
        cache: Optional[Cache] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        context : str, optional
            Text context shown together with the image(s), compressed during pre-filling.
        question : str, optional
            Question about the image(s)/context. Exclusive with `questions`.
        questions : list[str], optional
            Several questions answered independently on the same compressed cache.
        answer_prefix : str, optional
            Prefix added to the generated answer(s).
        method : CompressionMethod, optional
            The KV cache compression method applied during pre-filling.
        max_new_tokens : int, default=100
            Maximum number of tokens generated per answer.
        cache : Cache, optional
            Cache used for the forward pass. Defaults to a new DynamicCache.
        verbose : bool, default=False
            Print a summary with the measured KV cache sizes before/after compression.
        """
        assert question is None or questions is None, "Provide either question or questions, not both."
        preprocess_kwargs = {
            "context": context or "",
            "questions": questions or ([question] if question else [""]),
            "answer_prefix": answer_prefix,
        }
        forward_kwargs = {"method": method, "max_new_tokens": max_new_tokens, "cache": cache, "verbose": verbose}
        postprocess_kwargs = {"single_question": questions is None}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, images, context: str, questions: list[str], answer_prefix: str):
        """
        Build the multimodal chat prompt and split it into a context part (compressed)
        and per-question continuations (not compressed).
        """
        if not isinstance(images, (list, tuple)):
            images = [images]

        separator = "\n" + "#" * 13 + "\n"
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": context + separator})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        context_text, question_suffix = chat.split(separator)

        inputs = self.processor(text=[context_text], images=images, return_tensors="pt")

        questions_ids = [
            self.tokenizer.encode(
                question + question_suffix + answer_prefix, return_tensors="pt", add_special_tokens=False
            )
            for question in questions
        ]
        inputs["questions_ids"] = questions_ids
        return inputs

    def _forward(
        self,
        input_tensors,
        max_new_tokens: int = 100,
        method: Optional[CompressionMethod] = None,
        cache=None,
        verbose: bool = False,
    ):
        questions_ids = input_tensors.pop("questions_ids")
        input_tensors = {k: v.to(self.model.device) for k, v in input_tensors.items()}
        context_length = input_tensors["input_ids"].shape[1]

        if cache is None:
            cache = new_dynamic_cache(self.model)

        backbone = multimodal_backbone_of(self.model)

        # Pre-fill the multimodal context with compression. The multimodal backbone
        # computes and stores mrope position deltas when the model uses them.
        with compress(self.model, method, verbose=verbose):
            backbone(**input_tensors, past_key_values=cache, use_cache=True)

        logger.debug(f"Context length: {context_length}, compressed cache length: {cache.get_seq_length()}")

        # mrope positions of text tokens after the context continue at an offset stored
        # by the model during pre-filling (text positions advance by 1 per token)
        rope_deltas = getattr(backbone, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = torch.zeros(1, 1, dtype=torch.long, device=self.model.device)

        answers = []
        layer_indices = kv_cache_layer_indices(self.model)
        with per_layer_attention_masks(self.model):
            for question_ids in questions_ids:
                cache_snapshot = snapshot_cache(cache, layer_indices)
                answer = self.generate_answer(
                    question_ids.to(self.model.device), cache, context_length, rope_deltas, max_new_tokens
                )
                restore_cache(cache, cache_snapshot)
                answers.append(answer)
        return answers

    def _position_ids(self, start: int, length: int, rope_deltas: torch.Tensor) -> torch.Tensor:
        """3D mrope position ids for a text-only continuation starting at original position `start`."""
        positions = torch.arange(start, start + length, device=self.model.device)
        position_ids = positions[None, None, :].expand(3, rope_deltas.shape[0], length)
        return position_ids + rope_deltas[None].to(positions.device)

    def generate_answer(
        self,
        question_ids: torch.Tensor,
        cache: Cache,
        context_length: int,
        rope_deltas: torch.Tensor,
        max_new_tokens: int,
    ) -> str:
        """Greedy decoding of one answer on top of the (compressed) context cache."""
        q_len = question_ids.shape[1]
        position_ids = self._position_ids(context_length, q_len, rope_deltas)

        outputs = self.model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids, use_cache=True)
        generated_ids = [outputs.logits[0, -1].argmax()]

        eos_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]

        for i in range(max_new_tokens - 1):
            if generated_ids[-1].item() in eos_token_ids:
                break
            position_ids = self._position_ids(context_length + q_len + i, 1, rope_deltas)
            outputs = self.model(
                input_ids=generated_ids[-1].reshape(1, 1),
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
            )
            generated_ids.append(outputs.logits[0, -1].argmax())

        return self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)

    def postprocess(self, model_outputs, single_question: bool = True):
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}


def register_pipeline():
    """Register the pipeline so it can be created with transformers.pipeline(...)."""
    try:
        from transformers import AutoModelForImageTextToText

        PIPELINE_REGISTRY.register_pipeline(
            PIPELINE_NAME,
            pipeline_class=CompressedContextPipeline,
            pt_model=AutoModelForImageTextToText,
        )
    except Exception as exception:  # pragma: no cover
        logger.warning(f"Could not register the {PIPELINE_NAME} pipeline: {exception}")


register_pipeline()
