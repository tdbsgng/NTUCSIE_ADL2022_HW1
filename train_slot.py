import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
import random
import numpy as np
#from seqeval.metrics import classification_report
#from seqeval.scheme import IOB2
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
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

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS} 
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloaders= {
        split:DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional ,datasets[TRAIN].num_classes).to(args.device)
    '''
    ckpt = torch.load('./ckpt/slot/best_slot.pth')
    # load weights into model
    model.load_state_dict(ckpt)
    '''
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(model)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    epoch_count = 0
    best_acc = 0
    for epoch in epoch_pbar:
        correct = 0
        epoch_count += 1
        count = 0
        #y_pred=[]
        #y_true=[]
        # TODO: Training loop - iterate over train dataloader and update model weights
        
        for step,batch in enumerate(dataloaders[TRAIN]):
            optimizer.zero_grad()
            batch['tokens'] = batch['tokens'].to(args.device)
            batch['tags'] = batch['tags'].to(args.device)
            output_batch = model(batch)
            #print(output_batch.size(),batch['tags'].size())
            for index,data in enumerate(output_batch):
                loss = loss_fn(data,batch['tags'][index])
            loss.backward()
            optimizer.step()
            print(f'TRAIN Epoch : {epoch_count}/{args.num_epoch} Batch : {step+1}/{len(dataloaders[TRAIN])}, Loss : {loss}')
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            for step,batch in enumerate(dataloaders[DEV]):
                count += len(batch['tokens'])
                batch['tokens'] = batch['tokens'].to(args.device)
                batch['tags'] = batch['tags'].to(args.device)
                output_batch = model(batch)
                for index,data in enumerate(output_batch):
                    tags=[]
                    for output in data[:batch['len'][index]]:   
                        tags.append(int(torch.max(output,0).indices))
                    if tags == batch['tags'][index][:batch['len'][index]].tolist():
                        correct+=1
                    #y_pred.append(list(map(datasets[DEV].idx2label,tags)))
                    #y_true.append(list(map(datasets[DEV].idx2label,batch['tags'][index][:batch['len'][index]].tolist())))
            #print(classification_report(y_true,y_pred,scheme=IOB2, mode='strict'))
            print(f'EVAL Epoch : {epoch_count}/{args.num_epoch} Batch : {step+1}/{len(dataloaders[DEV])}, Acc : {correct}/{count}')
            if correct/count > best_acc:
                best_acc = correct/count
                if best_acc > 0.8:
                    torch.save(model.state_dict(), f'{args.ckpt_dir}/{args.hidden_size}_{args.num_layers}_{args.lr}_{args.batch_size}_{best_acc}.pth')
                    print(f'Epoch{epoch_count} model saved!')



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

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