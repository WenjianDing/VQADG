from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import re

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast


project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
vg_dir = dataset_dir.joinpath('COCO')
vg_feature_dir = vg_dir.joinpath('features')
vqadg_dir = dataset_dir.joinpath('vqav2')


class VQADGFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args
        self.mode = mode

        # Loading datasets to data
        if self.args.classifier:
            print('vqadg_data_vqav2.py classifier')
        else:
            print('vqadg_data_vqav2.py not classifier')
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)

        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        #self.answer_normalizer = VQAEvaluator()

        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            data_info_path = dataset_dir.joinpath(f'vqav2/{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                for _d in _data_info_dicts:
                    if source == 'vqav2_train':
                        self.img_ids_to_source[_d['img_id']] = 'train'
                    if source == 'vqav2_test':
                        self.img_ids_to_source[_d['img_id']] = 'test'
                    elif source == 'vqav2_val':
                        self.img_ids_to_source[_d['img_id']] = 'val'


                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes
        self.source_to_h5 = {
            'coco': dataset_dir.joinpath('COCO/features').joinpath('train2014_obj36.h5'),
        }
        # self.source_to_h5 = {
        #     'coco': dataset_dir.joinpath('COCO/features').joinpath('val2014_obj36.h5'),
        # }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            img_name = datum['img_name']
            out_dict['img_id'] = img_id
            #source = train/test/val
            source = self.img_ids_to_source[img_id]

            f = self.source_to_h5['coco']

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                #print('is instance')
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5['coco'] = f

            feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            try:
                f[f'{img_name}/features'].read_direct(feats)
            except KeyError:
                print('img_id', img_id)
                print(datum)
                exit()

            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_name}/img_h'][()]
            img_w = f[f'{img_name}/img_w'][()]
            boxes = f[f'{img_name}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes
            #print('feats', feats)
            #print('boses', boxes)


        ###### Text ######
        sent = datum['type']
        question = datum['question']
        multiple_choices = datum['multiple_choices']
        answer = datum['answer']
        image_id = datum['img_id']
        # qa = question + ';' + answer
        # print('question', question)
        # print('answer', answer)
        # print('qa',qa)
        #target = 'q:' + question + ';mc:' + str(multiple_choices) + ';a:' + answer
        target = 'q: ' + question + ';a: ' + answer + ';d: ' + str(multiple_choices)
        input_ids = self.tokenizer.encode(f'vqadg: {sent}', max_length=10, truncation=True)
        # 在pipeline中type,question,qa都要作为输入，这里max_length可以根据具体长度改变一下
        # input1_ids = self.tokenizer.encode(f'vqadg: {sent}', max_length=10, truncation=True)
        # input2_ids = self.tokenizer.encode(f'vqadg: {question}', max_length=20, truncation=True)
        # input3_ids = self.tokenizer.encode(f'vqadg: {qa}', max_length=30, truncation=True)

        # question_id = datum['qa_id']
        # out_dict['qa_id'] = question_id

        out_dict['sent'] = sent
        out_dict['image_id'] = image_id
        out_dict['raw_q'] = question
        out_dict['raw_a'] = answer
        out_dict['raw_mc'] = multiple_choices
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['target'] = target
        target_ids = self.tokenizer.encode(target, max_length=80, truncation=True)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        q_ids = self.tokenizer.encode('q: ' + question, max_length=20, truncation=True)
        out_dict['q_ids'] = torch.LongTensor(q_ids)
        out_dict['q_length'] = len(q_ids)
        a_ids = self.tokenizer.encode(';a: ' + answer, max_length=15, truncation=True)
        out_dict['a_ids'] = torch.LongTensor(a_ids)
        out_dict['a_length'] = len(a_ids)
        d_ids = self.tokenizer.encode(';d: ' + str(multiple_choices), max_length=45, truncation=True)
        out_dict['d_ids'] = torch.LongTensor(d_ids)
        out_dict['d_length'] = len(d_ids)


        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        S_W_L1 = max(entry['q_length'] for entry in batch)
        q_ids = torch.ones(B, S_W_L1, dtype=torch.long) * self.tokenizer.pad_token_id
        S_W_L2 = max(entry['a_length'] for entry in batch)
        a_ids = torch.ones(B, S_W_L2, dtype=torch.long) * self.tokenizer.pad_token_id
        S_W_L3 = max(entry['d_length'] for entry in batch)
        d_ids = torch.ones(B, S_W_L3, dtype=torch.long) * self.tokenizer.pad_token_id
        # print('qad L', S_W_L)
        if args.use_vision:
            V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        # if 'target' in batch[0]:
        #     # targets = []
        #     targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        sentences = []
        raw_questions = []
        raw_answers = []
        raw_mcs = []
        image_ids = []

        question_ids = []
        answers = []


        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            q_ids[i, :entry['q_length']] = entry['q_ids']
            a_ids[i, :entry['a_length']] = entry['a_ids']
            d_ids[i, :entry['d_length']] = entry['d_ids']

            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            # if 'target' in entry:
            #     targets[i] += entry['target']

            sentences.append(entry['sent'])
            raw_questions.append(entry['raw_q'])
            raw_answers.append(entry['raw_a'])
            raw_mcs.append(entry['raw_mc'])
            image_ids.append(entry['image_id'])
            #question_ids.append(entry['question_id'])
            #answers.append(entry['target'])


        batch_entry['input_ids'] = input_ids
        batch_entry['q_ids'] = q_ids
        batch_entry['a_ids'] = a_ids
        batch_entry['d_ids'] = d_ids
        batch_entry['q_lens'] = S_W_L1
        batch_entry['a_lens'] = S_W_L2
        batch_entry['d_lens'] = S_W_L3
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        # if 'target' in batch[0]:
        #     # targets = torch.stack(targets, dim=0)
        #     batch_entry['targets'] = targets

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
        batch_entry['sent'] = sentences
        batch_entry['raw_q'] = raw_questions
        batch_entry['raw_a'] = raw_answers
        batch_entry['raw_mc'] = raw_mcs
        batch_entry['image_ids'] = image_ids
        #batch_entry['question_ids'] = question_ids
        #batch_entry['answers'] = answers


        batch_entry['args'] = args
        batch_entry['task'] = 'vqadg'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)


    dataset = VQADGFineTuneDataset(
        split,
        raw_dataset=None,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)


    loader.task = 'vqadg'

    return loader

