"""
The module that contains utility functions and classes for models
"""
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import logging
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

def get_tf_model(
    base: str='bert'
) -> Tuple[object, ...]:
    """
    Get transformer model for NLP task.
    """
    logging.set_verbosity_error()
    if base == 'bert':
        pre_trained_model_name = 'bert-base-uncased'
        tf_tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
        tf_model = BertModel.from_pretrained(pre_trained_model_name, return_dict=True)
        config = BertConfig.from_pretrained(pre_trained_model_name)

    elif base == 'roberta':
        pre_trained_model_name = 'roberta-base'
        tf_tokenizer = RobertaTokenizer.from_pretrained(pre_trained_model_name)
        tf_model = RobertaModel.from_pretrained(pre_trained_model_name, return_dict=True)
        config = RobertaConfig.from_pretrained(pre_trained_model_name)

    else:
        tf_tokenizer = None
        tf_model = None
        config = None

    return tf_tokenizer, tf_model, config

class BaseModel(nn.Module):
    """
    The data structure for base models.
    NLP: Finetune transformer based models with 1-layer dnn.
    Tabular: 3-layer DNN with RELU activations.
    """
    def __init__(
        self,
        dataset: str='yelp',
        base: str='bert',
        input_dim: int=512,
        hidden_dim: int=768,
        tf_model: object=None,
        num_classes: int=2,
    ):
        super().__init__()
        self.model_name = 'base'

        self.dataset = dataset
        self.base = base

        if self.base == 'dnn':
            self.linear_base = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.transformer_model = tf_model

        self.base_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        if self.base == 'dnn':
            x, _ = inputs
            h = self.linear_base(x)
        else:
            input_ids, attention_mask, _ = inputs
            out = self.transformer_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            full_embedding = out['last_hidden_state']
            h = full_embedding[:,0]

        out = self.base_head(h)
        out = nn.functional.softmax(out, dim=1)
        out = torch.log(out)

        return out, h, _

class ConsequentEstimator(nn.Module):
    """
    The data structure for the consequent estimator.
    Predict mu, sigma, and coverage of given antecedent with 3-layer dnns.
    """
    def __init__(
        self,
        n_class: int=2,
        hidden_dim: int=768,
        atom_embedding: torch.Tensor=None,
    ):
        super().__init__()
        self.n_class = n_class
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.cp_te = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.atom_embedding=atom_embedding

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_class)
        )


        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_class)
        )

        self.coverage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        emb = torch.matmul(x, self.atom_embedding)
        out = self.cp_te(emb)
        out = torch.mean(out, dim=1)

        mu = F.softmax(self.mu_head(out), dim=-1)
        sigma = torch.exp(self.sigma_head(out).squeeze(dim=-1))
        coverage = torch.sigmoid(self.coverage_head(out).squeeze(dim=-1))

        return mu, sigma, coverage

class AtomSelector(nn.Module):
    """
    The data structure for the atom selector.
    Choose an atom for given instance.
    """
    def __init__(
        self,
        num_atoms: int=154,
        antecedent_len: int=2,
        hidden_dim: int=768,
        atom_embedding: torch.Tensor=None,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
        )
        self.dropout = nn.Dropout(0.1)
        self.gru_head = nn.Linear(hidden_dim, num_atoms)

        self.antecedent_len = antecedent_len
        self.atom_embedding = atom_embedding
        self.zero_v = None

    def filtered_softmax(
        self,
        x: torch.Tensor,
        x_: torch.Tensor,
        pos: int,
        pre_max_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conduct Gumbel-softmax for atoms that is satisfied by the instance.
        If an instance does not satisfy any atom, then choosen NULL atom.
        If NULL atom is once chosen, then the following atoms also become NULL atom.
        """
        assert len(x) == len(x_)
        if pos == 0:
            x_[torch.sum(x_, dim=-1) == 0, 0] = 1.0
        else:
            assert pre_max_index is not None
            x_[(pre_max_index==0), :] = 0.
            x_[:, 0] = 1.0

        x[~(x_.bool())] = float('-inf')
        x = F.gumbel_softmax(logits=x, tau=1, hard=True, dim=1)

        return x

    def forward(
        self,
        cls: torch.Tensor,
        x_: torch.Tensor,
    ) -> torch.Tensor:
        cls = cls.unsqueeze(dim=0).contiguous()
        cur_input = cls
        cur_h_0 = None

        atom_prob = []
        if self.zero_v is None:
            self.zero_v = torch.zeros(x_.shape).to(x_.device).long().detach()

        max_index = None
        for j in range(self.antecedent_len):
            if cur_h_0 is not None:
                _, h_n = self.gru(cur_input, cur_h_0)
            else:
                _, h_n = self.gru(cur_input)

            cur_h_0 = h_n
            h_n = h_n.squeeze(dim=0)
            h_n = self.dropout(h_n)
            out = self.gru_head(h_n)

            prob = self.filtered_softmax(out, x_, j, max_index)

            _, ind = torch.max(prob, dim=-1)
            ind = ind.unsqueeze(dim=1)
            src = self.zero_v
            x_ = torch.scatter(x_, dim=1, index=ind, src=src)

            atom_prob.append(prob)
            max_index = torch.max(prob, dim=-1)[1]

            atom_wsum = torch.mm(prob, self.atom_embedding.detach())
            cur_input = cls + atom_wsum.unsqueeze(dim=0)

        atom_prob = torch.stack(atom_prob, dim=1)

        return atom_prob


class AntecedentGenerator(BaseModel):
    """
    Data structure for antecedent generator.
    Sequentially chooses atoms with atom selectors,
    and obtain consequent by the consequent estimator.
    """
    def __init__(
        self,
        dataset: str='yelp',
        base: str='bert',
        antecedent_len: int=4,
        head: int=1,
        num_atoms: int=5001,
        input_dim: int=512,
        hidden_dim: int=768,
        num_classes: int=2,
        n_data: int=56000,
        atom_embedding: torch.Tensor=None,
        consequent_estimator: object=None,
        tf_model: object=None,
    ):
        super().__init__(
            dataset=dataset,
            base=base,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            tf_model=tf_model,
            num_classes=num_classes,
        )

        self.model_name = 'selor'
        self.rs_list = nn.ModuleList([
            AtomSelector(
                num_atoms=num_atoms,
                antecedent_len=antecedent_len,
                hidden_dim=hidden_dim,
                atom_embedding=atom_embedding,
            ) for i in range(head)
        ])

        self.tf_model = tf_model
        self.head = head
        self.antecedent_len = antecedent_len

        if consequent_estimator is None:
            self.consequent_estimator = None
        else:
            self.consequent_estimator = consequent_estimator

        self.n_class = num_classes
        self.num_atoms = num_atoms
        self.n_data = n_data

        self.base = base
        self.dataset = dataset

        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.atom_embedding = nn.Embedding(num_atoms, hidden_dim, _weight=atom_embedding)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        if self.base == 'dnn':
            x, x_ = inputs
            h = self.linear_base(x)
        else:
            input_ids, attention_mask, x_ = inputs
            out = self.tf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            full_embedding = out['last_hidden_state']
            h = full_embedding[:,0]

        atom_prob_list = []
        class_prob_list = []
        for i in range(self.head):
            # Choose atoms
            atom_prob = self.rs_list[i](h, x_.clone().detach())
            atom_prob_list.append(atom_prob)

            mu, _, coverage = self.consequent_estimator(atom_prob)
            n = coverage * self.n_data

            smooth = self.alpha * torch.reciprocal(n)
            smooth = smooth.unsqueeze(dim=-1).repeat(1, self.n_class)
            class_prob = torch.div(mu + smooth, 1 + self.n_class * smooth)

            class_prob_list.append(class_prob)

        mat_cp = torch.stack(class_prob_list, dim=1)
        mat_fp = torch.mean(mat_cp, dim=1)
        out = torch.log(mat_fp)

        return out, atom_prob_list, class_prob_list
