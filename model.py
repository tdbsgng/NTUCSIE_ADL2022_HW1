from typing import Dict
import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        
        self.lstm = torch.nn.LSTM(
            input_size = len(self.embeddings[0]), 
            hidden_size = self.hidden_size, 
            num_layers = self.num_layers, 
            dropout = self.dropout, 
            bidirectional = self.bidirectional,
            batch_first = True,
        )
        '''
        self.rnn = torch.nn.RNN(
            input_size = len(self.embeddings[0]), 
            hidden_size = self.hidden_size, 
            num_layers = self.num_layers, 
            dropout = self.dropout, 
            bidirectional = self.bidirectional,
            batch_first = True,
        )
        
        self.gru = torch.nn.GRU(
            input_size = len(self.embeddings[0]), 
            hidden_size = self.hidden_size, 
            num_layers = self.num_layers, 
            dropout = self.dropout, 
            bidirectional = self.bidirectional,
            batch_first = True,
        )
        '''
        self.linear = torch.nn.Linear(self.encoder_output_size, self.num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size
        raise NotImplementedError

    def forward(self, batch):#-> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch['text'] = self.embed(batch['text'])
        output_batch,(h,c) = self.lstm(batch['text'])
        #output_batch,h = self.gru(batch['text'])
        #output_batch,h = self.rnn(batch['text'])
        if self.bidirectional:
            h = torch.cat((h[-1], h[-2]), axis=-1) 
        else:
            h = h[-1]
        output_batch = self.linear(h)
        return output_batch
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch['tokens'] = self.embed(batch['tokens'])
        output_batch, (h,c) = self.lstm(batch['tokens'])
        #output_batch,h = self.gru(batch['tokens'])
        #output_batch,h = self.rnn(batch['tokens'])
        output_batch = self.linear(output_batch)
        return output_batch
        raise NotImplementedError
