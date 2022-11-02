import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

from torch.utils.data import DataLoader
from model import SeqClassifier
import random
import numpy as np

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloaders= {
        split:DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional ,datasets[TRAIN].num_classes).to(args.device)
    #print(model)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    '''
    ckpt = torch.load('./ckpt/intent/lstm/best.pth')
    # load weights into model
    model.load_state_dict(ckpt)
    '''
    print(model)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    epoch_count = 0
    best_acc = 0
    for epoch in epoch_pbar:
        correct = 0
        epoch_count += 1
        count = 0
        
        # TODO: Training loop - iterate over train dataloader and update model weights
        for step,batch in enumerate(dataloaders[TRAIN]):
            optimizer.zero_grad()
            batch['text'] = batch['text'].to(args.device)
            batch['intent'] = batch['intent'].to(args.device)
            output_batch = model(batch)
            loss = loss_fn(output_batch,batch['intent'])
            loss.backward()
            optimizer.step()
            print(f'TRAIN Epoch : {epoch_count}/{args.num_epoch} Batch : {step+1}/{len(dataloaders[TRAIN])}, Loss : {loss}')
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            for step,batch in enumerate(dataloaders[DEV]):
                count += len(batch['text'])
                batch['text'] = batch['text'].to(args.device)
                batch['intent'] = batch['intent'].to(args.device)
                output_batch = model(batch)
                loss = loss_fn(output_batch,batch['intent'])
                for index,output in enumerate(output_batch):
                    if torch.max(output,0).indices == batch['intent'][index]:
                        correct += 1
            print(f'EVAL Epoch : {epoch_count}/{args.num_epoch} Batch : {step+1}/{len(dataloaders[DEV])}, Loss : {loss} Acc : {correct}/{count}')
            if correct/count > best_acc:
                best_acc = correct/count
                if best_acc > 0.93:
                    torch.save(model.state_dict(), f'{args.ckpt_dir}/{args.hidden_size}_{args.num_layers}_{args.lr}_{args.batch_size}_{best_acc}.pth')
                    print(f'Epoch{epoch_count} model saved!')





def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
