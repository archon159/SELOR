import copy
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import Counter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Log(object):
    def __init__(self, log_path):
        self.flog = open(log_path, 'w')
        
    def write(self, s):
        print(s, file=self.flog, flush=True)
    
    def close(self):
        self.flog.close()
        
# The function for training cp predictor.
def pretrain(
    ce_model,
    train_dataloader,
    learning_rate,
    weight_decay,
    epochs,
    max_rule_len,
    n_data,
    weight_mu,
    weight_sigma,
    weight_coverage,
    gpu
):
    n_atom, hidden_dim = ce_model.atom_embedding.shape

    optimizer = torch.optim.AdamW(ce_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad()
    
    ce_model.train()
    
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        pbar = tqdm(range(len(train_dataloader)))

        total_losses = AverageMeter()
        mu_losses = AverageMeter()
        sigma_losses = AverageMeter()
        coverage_losses = AverageMeter()
        
        for d in train_dataloader:
            x, mu, sigma, n = d
            bsz = len(x)

            batch_size, rule_len = x.shape
            # We pad dummy if the rule length is smaller than max_rule_len
            x = F.pad(x, (0, max_rule_len - rule_len)).to(gpu)

            mu = mu.to(gpu)
            sigma = sigma.to(gpu)
            coverage = (n / n_data).to(gpu)

            mu_, sigma_, coverage_ = ce_model(x)
    
            # L2 distance as loss
            mu_loss = torch.mean(F.pairwise_distance(mu, mu_))
            sigma_loss = torch.mean(F.pairwise_distance(sigma, sigma_))
            coverage_loss = torch.mean(F.pairwise_distance(coverage.unsqueeze(dim=-1), coverage_.unsqueeze(dim=-1)))

            loss = weight_mu * mu_loss + weight_sigma * sigma_loss + weight_coverage * coverage_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_losses.update(loss.item(), n=bsz)
            mu_losses.update(mu_loss.item(), n=bsz)
            sigma_losses.update(sigma_loss.item(), n=bsz)
            coverage_losses.update(coverage_loss.item(), n=bsz)

            pbar.set_description(f'Train Loss: {total_losses.avg:.6f} ({mu_losses.avg:.6f}, {sigma_losses.avg:.6f}, {coverage_losses.avg:.6f})')
            pbar.update(1)
        pbar.close()

    return ce_model

def eval_pretrain(
    ce_model,
    test_dataloader,
#     atom_embedding,
    n_data,
    class_names,
#     args,
    gpu,
):
#     gpu = torch.device(f'cuda:{args.gpu}')
    n_atom, hidden_dim = ce_model.atom_embedding.shape
    ce_model.eval()
    
    mu_pred = []
    mu_answer = []
    sigma_pred = []
    sigma_answer = []
    coverage_pred = []
    coverage_answer = []
    
    for d in tqdm(test_dataloader):
        x, mu, sigma, n = d
        coverage = n / n_data

        batch_size, rule_len = x.shape
        x = F.pad(x, (0, 4 - rule_len))

        with torch.no_grad():
            mu_, sigma_, coverage_ = ce_model(x)

            mu_pred.extend(mu_)
            mu_answer.extend(mu)
            sigma_pred.extend(sigma_)
            sigma_answer.extend(sigma)
            coverage_pred.extend(coverage_)
            coverage_answer.extend(coverage)

    mu_pred = torch.stack(mu_pred).cpu()
    mu_answer = torch.stack(mu_answer).cpu()

    sigma_pred = torch.stack(sigma_pred).cpu()
    sigma_answer = torch.stack(sigma_answer).cpu()
        
    coverage_pred = torch.tensor(coverage_pred)
    coverage_answer = torch.tensor(coverage_answer)
    
    avg_mu_err = torch.mean(torch.abs(mu_pred - mu_answer)).item()
    avg_sigma_err = torch.mean(torch.abs(sigma_pred - sigma_answer)).item()
    avg_coverage_err = torch.mean(torch.abs(coverage_pred - coverage_answer)).item()
    
    # The label of predicted probability    
    q_mu_pred = torch.max(mu_pred, dim=1)[1]
    q_mu_answer = torch.max(mu_answer, dim=1)[1]

    classification_dict = classification_report(q_mu_answer, q_mu_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True)
    f1 = classification_dict['macro avg']['f1-score']
    
    return avg_mu_err, avg_sigma_err, avg_coverage_err, f1

def train_epoch(
    optimizer,
    model,
    loss_func,
    train_dataloader,
    gpu,
):
    
    train_loss = AverageMeter()
    pbar = tqdm(train_dataloader)

    model.train()
    for d in train_dataloader:
        inputs, y = d
        inputs = [i.to(gpu) for i in inputs]
        y = y.to(gpu)
        bsz = len(y)
        
        outputs, _, _ = model(
            inputs
        )

        loss = loss_func(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), n=bsz)

        pbar.update(1)
        pbar.set_description(f'Train Loss: {train_loss.avg:.3f}')
        # break
        
    pbar.close()

    return model, train_loss

def eval_epoch(
    model,
    loss_func,
    valid_dataloader,
    class_names,
    gpu,
):
    pbar = tqdm(valid_dataloader)
    model.eval()
    with torch.no_grad():
        valid_loss = AverageMeter()
        predictions = []
        target_probs = []
        answers = []
        for d in valid_dataloader:
            inputs, y = d
            inputs = [i.to(gpu) for i in inputs]
            y = y.to(gpu)
            batch_size = len(y)

            outputs, _, _ = model(
                inputs
            )
                
            loss = loss_func(outputs, y) 
            valid_loss.update(loss.item(), n=batch_size)
            
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            
            target_prob = torch.exp(outputs)[:, 1]
            target_probs.extend(target_prob)
            
            answers.extend(y)
            
            pbar.update(1)
            pbar.set_description(f'Valid Loss: {valid_loss.avg:.3f}')
            # break

        pbar.close()
        predictions = torch.stack(predictions).cpu().tolist()
        target_probs = torch.stack(target_probs).cpu().tolist()
        answers = torch.stack(answers).cpu().tolist()

        classification_dict = classification_report(answers, predictions, target_names=class_names, output_dict=True)
        roc_auc = roc_auc_score(answers, target_probs)
        precision, recall, thresholds = precision_recall_curve(answers, predictions, pos_label=1)
        pr_auc = auc(recall, precision)
        
        return classification_dict, roc_auc, pr_auc, valid_loss
    
def train(
    model,
    loss_func,
    train_dataloader,
    valid_dataloader,
    learning_rate,
    weight_decay,
    gamma,
    epochs,
    gpu,
    class_names,
    dir_path,
):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad()

    scheduler = ExponentialLR(optimizer, gamma=0.95)

    min_valid_loss = 100.0
    best_model_path = f'{dir_path}/model_best.pt'
    
    train_times = AverageMeter()
    valid_times = AverageMeter()
    
    log_path = f'{dir_path}/log'
    train_log = Log(log_path)
    
    train_log.write('Start Training')
    
    for epoch in range(epochs):
        train_log.write(f'Epoch {epoch}')
        
        train_start = time.time()
        model, train_loss = train_epoch(optimizer, model, loss_func, train_dataloader, gpu)
        train_end = time.time()
        train_time = train_end - train_start
        train_log.write(f'Training Time: {train_time:.3f} s')
        train_times.update(train_time)
        
        scheduler.step()
            
        valid_start = time.time()
        classification_dict, roc_auc, pr_auc, valid_loss = eval_epoch(model, loss_func, valid_dataloader, class_names, gpu)
        valid_end = time.time()
        valid_time = valid_end - valid_start
        train_log.write(f'Validation Time: {train_time:.3f} s')
        valid_times.update(valid_time)

        train_log.write(f'Train Loss: {train_loss.avg:.3f}')
        train_log.write(f'Valid Loss: {valid_loss.avg:.3f}')
        train_log.write(f'Valid Macro-F1: {classification_dict["macro avg"]["f1-score"]:.4f}')
        train_log.write(f'Valid ROC-AUC: {roc_auc:.4f}')
        train_log.write(f'Valid PR-AUC: {pr_auc:.4f}')
        train_log.write(f'\n')

        model_path = f'{dir_path}/model_epoch_{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        
        if valid_loss.avg < min_valid_loss:
            min_valid_loss = valid_loss.avg
            torch.save(model.state_dict(), best_model_path)

    train_log.write(f'Average Train Time: {train_times.avg:.3f} s')
    train_log.write(f'Average Valid Time: {valid_times.avg:.3f} s')
    train_log.close()
    
    return model
    
def eval_model(
    model,
    loss_func,
    test_dataloader,
    true_matrix,
    gpu,
    class_names,
    dir_path,
):
    eval_path = f'{dir_path}/model_eval'
    eval_log = Log(eval_path)
    
    pbar = tqdm(test_dataloader)
    model.eval()
    with torch.no_grad():
        test_loss = AverageMeter()
        predictions = []
        target_probs = []
        answers = []
        
        if model.model_name == 'selor':
            confidences = AverageMeter()
            consistencies = AverageMeter()
            duplicates = AverageMeter()
            coverages = AverageMeter()
            uniques = AverageMeter()
            lengths = []
            
        eval_start = time.time()
        for d in test_dataloader:
            inputs, y = d
            inputs = [i.to(gpu) for i in inputs]
            y = y.to(gpu)
            batch_size = len(y)

            # For base, we do not use atom_prob_list and cp_list
            outputs, atom_prob_list, cp_list = model(
                inputs
            )
            
            
            if model.model_name == 'selor':
                # Get the list of chosen atoms
                ind_list = []
                for atom_prob in atom_prob_list:
                    val, ind = torch.max(atom_prob, dim=-1)
                    rp = torch.prod(val, dim=-1)
                    ind_list.append(ind)

                    atoms = model.ae(ind)

                # Calculate Confidence
                cp_list = torch.stack(cp_list, dim=1)
                cp_mean_list = torch.mean(cp_list, dim=1)
                max_cp, _ = torch.max(cp_mean_list, dim=-1)
                confidence = torch.abs(max_cp - 0.5) * 2

                # Calculate Consistency
                _, num_head, n_class = cp_list.shape
                _, decision = torch.max(cp_list, dim=-1)
                cs = (decision == decision[:, 0].unsqueeze(dim=-1).repeat(1, num_head)).int()
                consistency = (torch.sum(cs, dim=-1) == num_head).float()

                ind_list = torch.stack(ind_list, dim=1)
                _, num_head, rule_length = ind_list.shape
                ind_list = ind_list.view(batch_size, num_head * rule_length)

                # Calculate the number of unique atoms
                unique = []
                for i in ind_list:
                    unique.append(len(set(i.tolist())))
                    
                # Calculate the length of rule
                length = torch.sum((ind_list != 0), dim=1)

                # Calculate the number of duplicated atoms
                token, _ = torch.mode(ind_list, dim=1)
                check = ind_list==token.unsqueeze(dim=-1)
                duplicate = torch.sum(check, dim=-1)
                
                # Calculate the coverage of the rule
                print(true_matrix.shape)
                assert(0)
                mat_rule_prob = torch.stack(atom_prob_list, dim=1)
                cover_rule_prob = torch.sum(mat_rule_prob, dim=2)
                cover_rule_prob = torch.matmul(cover_rule_prob, true_matrix)
                mat_satis = (cover_rule_prob == rule_length)
                mat_satis = torch.sum(mat_satis.float(), dim=-1)
                mat_coverage = mat_satis / n_data
                coverage = torch.mean(mat_coverage, dim=-1)
                
                confidences.update(confidence.mean().item(), batch_size)
                consistencies.update(consistency.mean().item(), batch_size)
                uniques.update(torch.tensor(unique).float().mean().item())
                lengths.extend(length.float())
                duplicates.update(duplicate.float().mean().item())
                coverages.update(coverage.float().mean().item())
            
            loss = loss_func(outputs,y)
            test_loss.update(loss.item(), batch_size)
            
            _, preds = torch.max(outputs, dim=1)
            target_prob = torch.exp(outputs)[:, 1]
            target_probs.extend(target_prob)
            
            predictions.extend(preds)
            answers.extend(y)
            
            pbar.update(1)
            pbar.set_description(f'Test Loss: {test_loss.avg:.3f}')
            # break
            
        pbar.close()
        eval_end = time.time()
        eval_time = eval_end - eval_start
                
        predictions = torch.stack(predictions).cpu().tolist()
        target_probs = torch.stack(target_probs).cpu().tolist()
        answers = torch.stack(answers).cpu().tolist()
        
        if model.model_name == 'selor':
            count_length = Counter(torch.stack(lengths).int().tolist())
            dist_length = {}
            for k, v in count_length.items():
                dist_length[k] = v / len(answers)

        eval_log.write(f'Avg Test Loss: {test_loss.avg:.4f}')
        eval_log.write(f'Evaluation Time: {eval_time:.3f} s')
        
        if model.model_name == 'selor':
            eval_log.write(f'Confidence: {confidences.avg:.4f}')
            eval_log.write(f'Consistency: {consistencies.avg:.4f}')
            eval_log.write(f'Duplicate: {duplicates.avg:.4f}')
            eval_log.write(f'Unique: {uniques.avg:.4f}')
            eval_log.write(f'Coverage: {coverages.avg:.4f}')
            eval_log.write(f'Length')
            for i in range(rule_length + 1):
                if i in dist_length:
                    eval_log.write(f'{i}: {dist_length[i]:.4f}')
        
        roc_auc = roc_auc_score(answers, target_probs)
        precision, recall, thresholds = precision_recall_curve(answers, predictions, pos_label=1)
        pr_auc = auc(recall, precision)
        
        eval_log.write('Prediction Performance:')
        eval_log.write(classification_report(answers, predictions, target_names=class_names, digits=4))
        eval_log.write(f'ROC-AUC: {roc_auc:.4f}')
        eval_log.write(f'PR-AUC: {pr_auc:.4f}')
        
    eval_log.close()
        
def get_base_embedding(
    model,
    train_dataloader,
    gpu,
):
    assert(model.model_name == 'base')
    
    h_list = []
    
    model.eval()
    with torch.no_grad():
        for d in tqdm(train_dataloader):
            inputs, y = d
            inputs = [i.to(gpu) for i in inputs]
            y = y.to(gpu)
            batch_size = len(y)

            outputs, h, _ = model(
                inputs
            )

            h_list.append(h.cpu())
    embeddings = torch.cat(h_list, dim=0)
    return embeddings
