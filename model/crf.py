# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 10:39 PM

from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F


class ConfigurationError(Exception):
    """
    Custom exception to replace allennlp.common.checks.ConfigurationError.
    """
    pass


def logsumexp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the log-sum-exp of the input tensor along the specified dimension.
    """
    max_score, _ = tensor.max(dim)
    return max_score + (tensor - max_score.unsqueeze(dim)).exp().sum(dim).log()


def viterbi_decode(tag_sequence: torch.Tensor, transitions: torch.Tensor) -> Tuple[List[int], float]:
    """
    Decodes the best path using the Viterbi algorithm.
    """
    sequence_length, num_tags = tag_sequence.size()
    path_scores = tag_sequence[0]
    backpointers = []

    for i in range(1, sequence_length):
        broadcast_path_scores = path_scores.unsqueeze(1)
        scores = broadcast_path_scores + transitions + tag_sequence[i].unsqueeze(0)
        max_scores, max_score_indices = scores.max(dim=0)
        path_scores = max_scores
        backpointers.append(max_score_indices)

    best_last_tag = path_scores.argmax().item()
    best_path = [best_last_tag]
    for backpointer in reversed(backpointers):
        best_last_tag = backpointer[best_last_tag].item()
        best_path.append(best_last_tag)

    best_path.reverse()
    best_score = path_scores[best_path[-1]].item()
    return best_path, best_score


def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions.
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str) -> bool:
    """
    Determines if a transition is allowed under the given constraint type.
    """
    if to_tag == "START" or from_tag == "END":
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ('O', 'B', 'U')
        if to_tag == "END":
            return from_tag in ('O', 'L', 'U')
        return any([
            from_tag in ('O', 'L', 'U') and to_tag in ('O', 'B', 'U'),
            from_tag in ('B', 'I') and to_tag in ('I', 'L') and from_entity == to_entity
        ])
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ('O', 'B')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
            to_tag in ('O', 'B'),
            to_tag == 'I' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ('O', 'I')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
            to_tag in ('O', 'I'),
            to_tag == 'B' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ('B', 'S')
        if to_tag == "END":
            return from_tag in ('E', 'S')
        return any([
            to_tag in ('B', 'S') and from_tag in ('E', 'S'),
            to_tag == 'M' and from_tag in ('B', 'M') and from_entity == to_entity,
            to_tag == 'E' and from_tag in ('B', 'M') and from_entity == to_entity,
        ])
    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")


class ConditionalRandomField(torch.nn.Module):
    """
    Conditional Random Field implementation without allennlp dependencies.
    """

    def __init__(self, num_tags: int, constraints: List[Tuple[int, int]] = None, include_start_end_transitions: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        if constraints is None:
            constraint_mask = torch.ones(num_tags + 2, num_tags + 2)
        else:
            constraint_mask = torch.zeros(num_tags + 2, num_tags + 2)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_tags = logits.size()
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            inner = broadcast_alpha + emit_scores + transition_scores
            alpha = (logsumexp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        return logsumexp(stops, dim=1)

    def _joint_likelihood(self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        batch_size, sequence_length, _ = logits.data.shape
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1)).squeeze()
        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.Tensor) -> List[Tuple[List[int], float]]:
        _, max_seq_length, num_tags = logits.size()
        logits, mask = logits.data, mask.data

        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.full((num_tags + 2, num_tags + 2), -10000.0)

        constrained_transitions = (
            self.transitions * self._constraint_mask[:num_tags, :num_tags] +
            -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        )
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data +
                -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data +
                -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        tag_sequence = torch.full((max_seq_length + 2, num_tags + 2), -10000.0)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)

            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            viterbi_path, viterbi_score = viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score))

        return best_paths