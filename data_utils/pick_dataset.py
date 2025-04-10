# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/9/2020 9:16 PM
import glob
import os
from typing import *
from pathlib import Path
import warnings
import random
from overrides import overrides

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from . import documents
from .documents import Document
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls, entities_vocab_cls


class PICKDataset(Dataset):

    def __init__(self, files_name: str = None,
                 boxes_and_transcripts_folder: str = 'boxes_and_transcripts',
                 images_folder: str = 'images',
                 entities_folder: str = 'entities',
                 iob_tagging_type: str = 'box_and_within_box_level',
                 resized_image_size: Tuple[int, int] = (480, 960),
                 keep_ratio: bool = True,
                 ignore_error: bool = False,
                 training: bool = True
                 ):
        '''

        :param files_name: containing training and validation samples list file.
        :param boxes_and_transcripts_folder: gt or ocr result containing transcripts, boxes and box entity type (optional).
        :param images_folder: whole images file folder
        :param entities_folder: exactly entity type and entity value of documents, containing json format file
        :param iob_tagging_type: 'box_level', 'document_level', 'box_and_within_box_level'
        :param resized_image_size: resize whole image size, (w, h)
        :param keep_ratio: TODO implement this parames
        :param ignore_error:
        :param training: True for train and validation mode, False for test mode. True will also load labels,
        and files_name and entities_file must be set.
        '''
        super().__init__()
        self._image_ext = None
        self._ann_ext = None
        self.iob_tagging_type = iob_tagging_type
        self.keep_ratio = keep_ratio
        self.ignore_error = ignore_error
        self.training = training
        assert resized_image_size and len(resized_image_size) == 2, 'resized image size not be set.'
        self.resized_image_size = tuple(resized_image_size)  # (w, h)

        if self.training:  # used for train and validation mode
            self.files_name = Path(files_name)
            self.data_root = self.files_name.parent
            self.boxes_and_transcripts_folder: Path = self.data_root.joinpath(boxes_and_transcripts_folder)
            self.images_folder: Path = self.data_root.joinpath(images_folder)
            self.entities_folder: Path = self.data_root.joinpath(entities_folder)
            if self.iob_tagging_type != 'box_level':
                if not self.entities_folder.exists():
                    raise FileNotFoundError('Entity folder is not exist!')
        else:  # used for test mode
            self.boxes_and_transcripts_folder: Path = Path(boxes_and_transcripts_folder)
            self.images_folder: Path = Path(images_folder)

        if not (self.boxes_and_transcripts_folder.exists() and self.images_folder.exists()):
            raise FileNotFoundError('Not contain boxes_and_transcripts floader {} or images folder {}.'
                                    .format(self.boxes_and_transcripts_folder.as_posix(),
                                            self.images_folder.as_posix()))
        if self.training:
            self.files_list = pd.read_csv(self.files_name.as_posix(), header=None,
                                          names=['index', 'document_class', 'file_name'],
                                          dtype={'index': int, 'document_class': str, 'file_name': str})
        else:
            self.files_list = list(self.boxes_and_transcripts_folder.glob('*.tsv'))

    def __len__(self):
        return len(self.files_list)

    def get_image_file(self, basename):
        """
        Return the complete name (fill the extension) from the basename.
        """
        if self._image_ext is None:
            filename = list(self.images_folder.glob(f'**/{basename}.*'))[0]
            self._image_ext = os.path.splitext(filename)[1]

        return self.images_folder.joinpath(basename + self._image_ext)

    def get_ann_file(self, basename):
        """
        Return the complete name (fill the extension) from the basename.
        """
        if self._ann_ext is None:
            filename = list(self.boxes_and_transcripts_folder.glob(f'**/{basename}.*'))[0]
            self._ann_ext = os.path.splitext(filename)[1]

        return self.boxes_and_transcripts_folder.joinpath(basename + self._ann_ext)

    @overrides
    def __getitem__(self, index):

        if self.training:
            dataitem: pd.Series = self.files_list.iloc[index]
            # config file path
            boxes_and_transcripts_file = self.get_ann_file(Path(dataitem['file_name']).stem)
            image_file = self.get_image_file(Path(dataitem['file_name']).stem)
            entities_file = self.entities_folder.joinpath(Path(dataitem['file_name']).stem + '.txt')
            # documnets_class = dataitem['document_class']
        else:
            boxes_and_transcripts_file = self.get_ann_file(Path(self.files_list[index]).stem)
            image_file = self.get_image_file(Path(self.files_list[index]).stem)

        if not boxes_and_transcripts_file.exists() or not image_file.exists():
            if self.ignore_error and self.training:
                warnings.warn('{} is not exist. get a new one.'.format(boxes_and_transcripts_file))
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Sample: {} not exist.'.format(boxes_and_transcripts_file.stem))

        try:
            # TODO add read and save cache function, to speed up data loaders

            if self.training:
                document = documents.Document(boxes_and_transcripts_file, image_file, self.resized_image_size,
                                              self.iob_tagging_type, entities_file, training=self.training)
            else:
                document = documents.Document(boxes_and_transcripts_file, image_file, self.resized_image_size,
                                              image_index=index, training=self.training)
            return document
        except Exception as e:
            if self.ignore_error:
                warnings.warn('loading samples is occurring error, try to regenerate a new one.')
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Error occurs in image {}: {}'.format(boxes_and_transcripts_file.stem, e.args))


class BatchCollateFn(object):
    '''
    padding input (List[Example]) with same shape, then convert it to batch input.
    '''

    def __init__(self, training: bool = True):
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.training = training

    def __call__(self, batch_list: List[Document]):

        for i, doc in enumerate(batch_list):
            print(f'[{i}] {doc.image_filename} boxes_num: {doc.boxes_num}, transcript_len: {doc.transcript_len}')

        # In toàn bộ các thuộc tính của từng đối tượng Document
        for i, x in enumerate(batch_list):
            print(f"\n--- Document {i} ---")
            for attr, value in vars(x).items():  # Duyệt qua tất cả các thuộc tính
                print(f"{attr}: {value}")

        # dynamic calculate max boxes number of batch,
        max_boxes_num_batch = max([x.boxes_num for x in batch_list])
        max_transcript_len = max([x.transcript_len for x in batch_list])

        # Tiếp tục các xử lý khác...
