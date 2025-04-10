        
# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/11/2020 10:39 PM

from typing import *
from collections import defaultdict
import torch
import numpy as np

'''
Custom implementation to replace torchtext and allennlp:
    * Add accuracy measure mEA (mean Entity Accuracy)
    * Rename precision, recall, f1 to mEP, mER, mEF
    * Numerical stability
'''

TAGS_TO_SPANS_FUNCTION_TYPE = Callable[
    [List[str], Optional[List[str]]], List[Tuple[str, Tuple[int, int]]]]  # pylint: disable=invalid-name


def bio_tags_to_spans(tag_sequence: List[str], ignore_classes: Optional[List[str]] = None) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Converts a sequence of BIO tags into spans.
    """
    spans = []
    span_start = None
    span_label = None
    for index, tag in enumerate(tag_sequence):
        if tag.startswith("B-"):
            if span_label is not None:
                spans.append((span_label, (span_start, index - 1)))
            span_label = tag[2:]
            span_start = index
        elif tag == "O":
            if span_label is not None:
                spans.append((span_label, (span_start, index - 1)))
                span_label = None
                span_start = None
    if span_label is not None:
        spans.append((span_label, (span_start, len(tag_sequence) - 1)))
    if ignore_classes:
        spans = [span for span in spans if span[0] not in ignore_classes]
    return spans


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the lengths of sequences in a batch based on a binary mask.
    """
    return mask.sum(dim=1)


class SpanBasedF1Measure:
    """
    Implements span-based precision, recall, and F1 metrics for a BIO tagging scheme.
    """

    def __init__(self,
                 vocab: Dict[str, int] = None,
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO",
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None) -> None:
        """
        Parameters
        ----------
        vocab : ``Dict[str, int]``, required.
            A dictionary containing the tag namespace.
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
        label_encoding : ``str``, optional (default = "BIO")
            The encoding used to specify label span endpoints in the sequence.
            Valid options are "BIO", "IOB1", "BIOUL" or "BMES".
        tags_to_spans_function: ``Callable``, optional (default = ``None``)
            If ``label_encoding`` is ``None``, ``tags_to_spans_function`` will be
            used to generate spans.
        """
        if label_encoding and tags_to_spans_function:
            raise ValueError(
                'Both label_encoding and tags_to_spans_function are provided. '
                'Set "label_encoding=None" explicitly to enable tags_to_spans_function.'
            )
        if label_encoding:
            if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES"]:
                raise ValueError("Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'.")
        elif tags_to_spans_function is None:
            raise ValueError(
                'At least one of the (label_encoding, tags_to_spans_function) should be provided.'
            )

        self._label_encoding = label_encoding
        self._tags_to_spans_function = tags_to_spans_function or bio_tags_to_spans
        self._label_vocabulary = vocab
        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
        self._total: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask).long()
        argmax_predictions = predictions.argmax(dim=-1)

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :sequence_lengths[i]]
            sequence_gold_label = gold_labels[i, :sequence_lengths[i]]

            predicted_string_labels = [self._label_vocabulary[idx] for idx in sequence_prediction.tolist()]
            gold_string_labels = [self._label_vocabulary[idx] for idx in sequence_gold_label.tolist()]

            predicted_spans = self._tags_to_spans_function(predicted_string_labels, self._ignore_classes)
            gold_spans = self._tags_to_spans_function(gold_string_labels, self._ignore_classes)

            for span in gold_spans:
                self._total[span[0]] += 1

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1

            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing the span-based metrics:
        precision : float
        recall : float
        f1-measure : float
        """
        all_metrics = {}
        for tag in self._total.keys():
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag],
                self._false_positives[tag],
                self._false_negatives[tag]
            )
            all_metrics[f"mEP-{tag}"] = precision
            all_metrics[f"mER-{tag}"] = recall
            all_metrics[f"mEF-{tag}"] = f1_measure

        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
        self._total = defaultdict(int)

    @staticmethod
    def detach_tensors(*tensors):
        return (tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors)