# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:26 PM

from collections import Counter
from pathlib import Path

from . import entities_list


class CustomVocab:
    def __init__(self, classes, specials=['<pad>', '<unk>'], specials_first=True):
        '''
        :param classes: list or str, key string or entity list
        :param specials: list, special tokens (default: ['<pad>', '<unk>'])
        :param specials_first: bool, whether to add specials at the beginning
        '''
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        elif isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read().strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes
        else:
            raise ValueError("Unsupported type for classes: {}".format(type(classes)))

        # Count occurrences of each class
        counter = Counter(cls_list)

        # Add special tokens
        if specials_first:
            self.itos = specials + list(counter.keys())
        else:
            self.itos = list(counter.keys()) + specials

        # Create stoi (string-to-index) mapping
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


def entities2iob_labels(entities: list):
    '''
    Get all IOB string labels by entities.
    :param entities: list of entity names
    :return: list of IOB labels
    '''
    tags = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    return tags


# Initialize vocabularies
keys_vocab_cls = CustomVocab(Path(__file__).parent.joinpath('keys.txt'), specials_first=False)
iob_labels_vocab_cls = CustomVocab(entities2iob_labels(entities_list.Entities_list), specials_first=False)
entities_vocab_cls = CustomVocab(entities_list.Entities_list, specials_first=False)