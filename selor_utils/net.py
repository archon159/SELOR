import torch
import torch.nn as nn
import torch.nn.functional as F

def get_tf_model(
    base='bert'
):
    if base == 'bert':
        from transformers import BertModel, BertTokenizer, BertConfig
        
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        tf_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        tf_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
        config = BertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
    elif base == 'roberta':
        from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
        
        PRE_TRAINED_MODEL_NAME = 'roberta-base'
        tf_tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        tf_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
        config = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
    else:
        tf_tokenizer = None
        tf_model = None
        config = None
        
    return tf_tokenizer, tf_model, config

class BaseModel(nn.Module):
    def __init__(
        self,
        dataset='yelp',
        base='bert',
        input_dim=512,
        hidden_dim=int,
        tf_model=None,
        num_classes=2,
        dropout=0,
    ):
        super(BaseModel, self).__init__()
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
        
    def forward(self, inputs):
        if self.base == 'dnn':
            x, _ = inputs
            h = self.linear_base(x)
        else:
            input_ids, attention_mask, _ = inputs
            batch_size, max_len = input_ids.shape
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
    def __init__(
        self,
        n_class=2,
        hidden_dim=768,
        atom_embedding=None,
        args=None,
    ):
        super(ConsequentEstimator, self).__init__()
        self.args = args
        self.n_class = n_class
        self.dataset = args.dataset
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.cp_te = nn.TransformerEncoder(encoder_layer, num_layers=6)

        if self.dataset=='yelp':
            self.mu_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.mu_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_class)
            )
            
        if self.dataset=='yelp':
            self.sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
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
        e,
    ):
        out = self.cp_te(e)
        out = torch.mean(out, dim=1)

        if self.dataset=='yelp':
            mu = torch.sigmoid(self.mu_head(out).squeeze(dim=-1))
        else:
            mu = F.softmax(self.mu_head(out), dim=-1)
            
        sigma = torch.exp(self.sigma_head(out).squeeze(dim=-1))
        coverage = torch.sigmoid(self.coverage_head(out).squeeze(dim=-1))
        
        return mu, sigma, coverage
    
class AtomSelector(nn.Module):
    def __init__(
        self,
        num_atoms=154,
        rule_len=2,
        hidden_dim=768,
        ae=None,
    ):
        super(AtomSelector, self).__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
        )
        self.dropout = nn.Dropout(0.1)
        self.gru_head = nn.Linear(hidden_dim, num_atoms)
        
        self.rule_len = rule_len
        self.ae = ae
        self.zero_v = None
    
    def filtered_softmax(self, x, x_, pos, pre_max_index):
        assert(len(x) == len(x_))
        if pos == 0:
            x_[torch.sum(x_, dim=-1) == 0, 0] = 1.0
        else:
            assert(pre_max_index != None)
            x_[(pre_max_index==0), :] = 0.
            x_[:, 0] = 1.0
            
        x[~(x_.bool())] = float('-inf')
        x = F.gumbel_softmax(logits=x, tau=1, hard=True, dim=1)

        return x
    
    def forward(
        self,
        cls,
        x_,
    ):
        cls = cls.unsqueeze(dim=0).contiguous()
        cur_input = cls
        cur_h_0 = None
        
        atom_prob = []
        if self.zero_v == None:
            self.zero_v = torch.zeros(x_.shape).to(x_.device).long().detach()
        
        max_index = None
        for j in range(self.rule_len):
            if cur_h_0 != None:
                _, h_n = self.gru(cur_input, cur_h_0)
            else:
                _, h_n = self.gru(cur_input)

            cur_h_0 = h_n
            h_n = h_n.squeeze(dim=0)
            h_n = self.dropout(h_n)
            out = self.gru_head(h_n)

            prob = self.filtered_softmax(out, x_, j, max_index)

            val, ind = torch.max(prob, dim=-1)
            ind = ind.unsqueeze(dim=1)
            src = self.zero_v
            x_ = torch.scatter(x_, dim=1, index=ind, src=src)

            atom_prob.append(prob)
            max_index = torch.max(prob, dim=-1)[1]
            
            atom_wsum = torch.mm(prob, self.ae.weight.detach())
            cur_input = cls + atom_wsum.unsqueeze(dim=0)
            
        atom_prob = torch.stack(atom_prob, dim=1)
        return atom_prob
    
    
class RuleGenerator(BaseModel):
    def __init__(
        self,
        rule_len=4,
        head=1,
        num_atoms=5001,
        input_dim=512,
        hidden_dim=768,
        vocab_size=0,
        num_classes=2,
        n_data=56000,
        atom_embedding=None,
        consequent_estimator=None,
        padding_idx=0,
        tf_model=None,
        args=None,
    ):
        super(RuleGenerator, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_classes=num_classes,
            padding_idx=padding_idx,
            tf_model=tf_model,
            args=args,
        )
        
        self.model_name = 'rule_gen'
        self.ae = nn.Embedding(num_atoms, hidden_dim, _weight=atom_embedding)
        self.args = args
        self.rs_list = nn.ModuleList([
            AtomSelector(
                num_atoms=num_atoms,
                rule_len=rule_len,
                hidden_dim=hidden_dim,
                ae=self.ae,
            ) for i in range(head)
        ])
        
        self.tf_model = tf_model
        self.head = head
        self.rule_len = rule_len
        
        if consequent_estimator == None:
            self.consequent_estimator = None
        else:
            self.consequent_estimator = consequent_estimator
            
        self.n_class = num_classes
        self.num_atoms = num_atoms
        self.n_data = n_data
        
        self.base = args.base_model
        self.dataset = args.dataset
        
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)
        
    def forward(
        self,
        inputs
    ):
        if self.base == 'dnn':
            x, x_ = inputs
            batch_size, _ = x.shape
            h = self.linear_base(x)
        else:
            input_ids, attention_mask, x_ = inputs
            batch_size, max_len = input_ids.shape
            out = self.tf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            full_embedding = out['last_hidden_state']
            h = full_embedding[:,0]
        
        atom_prob_list = []
        cp_list = []
        for i in range(self.head):
            # Choose atoms
            atom_prob = self.rs_list[i](h, x_.clone().detach())
            atom_prob_list.append(atom_prob)
            
            atoms = torch.matmul(atom_prob, self.ae.weight.detach())
            mu, sigma, coverage = self.consequent_estimator(atoms)
            n = coverage * self.n_data

            if self.consequent_estimator.dataset=='yelp':
                pp = torch.div((mu + self.alpha * torch.reciprocal(n)), (1 + 2 * self.alpha * torch.reciprocal(n)))
                cp = torch.stack((1 - pp, pp), dim=-1)
            else:
                batch_size, n_class = mu.shape
                assert(self.n_class == n_class)
                sf = self.alpha * torch.reciprocal(n)
                sf = sf.unsqueeze(dim=-1).repeat(1, n_class)
                cp = torch.div(mu + sf, 1 + 2 * sf)
                
            cp_list.append(cp)

        mat_cp = torch.stack(cp_list, dim=1)
        mat_fp = torch.mean(mat_cp, dim=1)
        out = torch.log(mat_fp)
        
        return out, atom_prob_list, cp_list