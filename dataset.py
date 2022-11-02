from __future__ import barry_as_FLUFL
from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len
import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batch = {}
        batch['text'] = torch.tensor(self.vocab.encode_batch([data['text'].split() for data in samples], self.max_len))
        #print(batch['text'][0][0])
        batch['id'] = [data['id'] for data in samples]
        if 'intent' in samples[0].keys(): 
            batch['intent'] = torch.tensor([self.label2idx(data['intent']) for data in samples])
        return batch
        raise NotImplementedError
        


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        batch = {}
        batch['tokens'] = torch.tensor(self.vocab.encode_batch([data['tokens'] for data in samples], self.max_len))
        batch['id'] = [data['id'] for data in samples]
        batch['len'] = [len(data['tokens']) for data in samples]
        if 'tags' in samples[0].keys():
            batch['tags'] = torch.tensor(pad_to_len([[self.label2idx(tag) for tag in data['tags']] for data in samples],self.max_len,0))
        return batch
        raise NotImplementedError
