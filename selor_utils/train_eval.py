"""
The module that contains utility functions related to model training and evaluation
"""
import time
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
from .dataset import get_single_input
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(
        self
    ):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        val: float,
        n: int=1
    ):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pretrain(
    ce_model: object,
    train_dataloader: object,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    max_antecedent_len: int,
    n_data: int,
    w_mu: torch.Tensor,
    w_sigma: torch.Tensor,
    w_coverage: torch.Tensor,
    gpu: torch.device
) -> object:
    """
    Pretrains the consequent estimator with given train dataloader.
    """
    n_atom, _ = ce_model.atom_embedding.shape

    optimizer = AdamW(
        ce_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    optimizer.zero_grad()

    ce_model.train()

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        pbar = tqdm(range(len(train_dataloader)))

        total_losses = AverageMeter()
        mu_losses = AverageMeter()
        sigma_losses = AverageMeter()
        coverage_losses = AverageMeter()

        for batch in train_dataloader:
            x, mu, sigma, n = batch
            bsz, antecedent_len = x.shape
            # We pad dummy if the antecedent length is smaller than antecedent_len
            x = F.pad(x, (0, max_antecedent_len - antecedent_len)).to(gpu)
            x = F.one_hot(x, n_atom).float()

            mu = mu.to(gpu)
            sigma = sigma.to(gpu)
            coverage = (n / n_data).to(gpu)

            mu_, sigma_, coverage_ = ce_model(x)

            # L2 distance as loss
            mu_loss = torch.mean(
                F.pairwise_distance(
                    mu,
                    mu_
                )
            )
            sigma_loss = torch.mean(
                F.pairwise_distance(
                    sigma,
                    sigma_
                )
            )
            coverage_loss = torch.mean(
                F.pairwise_distance(
                    coverage.unsqueeze(dim=-1),
                    coverage_.unsqueeze(dim=-1)
                )
            )

            loss = w_mu * mu_loss + w_sigma * sigma_loss + w_coverage * coverage_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_losses.update(loss.item(), n=bsz)
            mu_losses.update(mu_loss.item(), n=bsz)
            sigma_losses.update(sigma_loss.item(), n=bsz)
            coverage_losses.update(coverage_loss.item(), n=bsz)

            desc = f'Train Loss: {total_losses.avg:.6f} '
            desc += f'({mu_losses.avg:.6f}, {sigma_losses.avg:.6f}, {coverage_losses.avg:.6f})'
            pbar.set_description(desc)
            pbar.update(1)
        pbar.close()

    return ce_model

def eval_pretrain(
    ce_model: object,
    test_dataloader: object,
    max_antecedent_len: int,
    n_data: int,
    class_names: List[str],
    gpu: torch.device,
) -> Tuple[float, ...]:
    """
    Evaluate the consequent estimator with given test dataloader
    """
    n_atom, _ = ce_model.atom_embedding.shape
    ce_model.eval()

    mu_pred = []
    mu_answer = []
    sigma_pred = []
    sigma_answer = []
    coverage_pred = []
    coverage_answer = []

    for batch in tqdm(test_dataloader):
        x, mu, sigma, n = batch
        coverage = n / n_data

        _, antecedent_len = x.shape
        x = F.pad(x, (0, max_antecedent_len - antecedent_len)).to(gpu)
        x = F.one_hot(x, n_atom).float()

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

    coverage_pred = torch.Tensor(coverage_pred)
    coverage_answer = torch.Tensor(coverage_answer)

    avg_mu_err = torch.mean(torch.abs(mu_pred - mu_answer)).item()
    avg_sigma_err = torch.mean(torch.abs(sigma_pred - sigma_answer)).item()
    avg_coverage_err = torch.mean(torch.abs(coverage_pred - coverage_answer)).item()

    # The label of predicted probability
    q_mu_pred = torch.max(mu_pred, dim=1)[1]
    q_mu_answer = torch.max(mu_answer, dim=1)[1]

    classification_dict = classification_report(
        q_mu_answer,
        q_mu_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    f1_score = classification_dict['macro avg']['f1-score']

    return avg_mu_err, avg_sigma_err, avg_coverage_err, f1_score

def train_epoch(
    optimizer: object,
    model: object,
    loss_func: object,
    train_dataloader: object,
    gpu: torch.device,
) -> Tuple[object, ...]:
    """
    Train the model for an epoch
    """
    train_loss = AverageMeter()
    pbar = tqdm(train_dataloader)

    model.train()
    for batch in train_dataloader:
        inputs, y = batch
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

    pbar.close()

    return model, train_loss

def eval_epoch(
    model: object,
    loss_func: object,
    valid_dataloader: object,
    class_names: List[str],
    gpu: torch.device,
) -> Tuple[dict, float, float, object]:
    """
    Evaluate the model for an epoch
    """
    pbar = tqdm(valid_dataloader)
    model.eval()
    with torch.no_grad():
        valid_loss = AverageMeter()
        predictions = []
        target_probs = []
        answers = []
        for batch in valid_dataloader:
            inputs, y = batch
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

        pbar.close()
        predictions = torch.stack(predictions).cpu().tolist()
        target_probs = torch.stack(target_probs).cpu().tolist()
        answers = torch.stack(answers).cpu().tolist()

        classification_dict = classification_report(
            answers,
            predictions,
            target_names=class_names,
            output_dict=True
        )
        roc_auc = roc_auc_score(answers, target_probs)
        precision, recall, _ = precision_recall_curve(
            answers,
            predictions,
            pos_label=1
        )
        pr_auc = auc(recall, precision)

        return classification_dict, roc_auc, pr_auc, valid_loss

def train(
    model: object,
    loss_func: object,
    train_dataloader: object,
    valid_dataloader: object,
    learning_rate: float,
    weight_decay: float,
    gamma: float,
    epochs: int,
    gpu: torch.device,
    class_names: List[str],
    dir_path: Path,
) -> object:
    """
    Train the model and evaluate for entire epochs with given train and valid dataloader.
    """
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    optimizer.zero_grad()

    scheduler = ExponentialLR(optimizer, gamma=gamma)

    min_valid_loss = 100.0
    best_model_path = dir_path / 'model_best.pt'

    train_times = AverageMeter()
    valid_times = AverageMeter()

    train_log = logging.getLogger()
    train_log.setLevel(logging.INFO)
    train_file_handler = logging.FileHandler(str(dir_path / 'log'), mode='w')
    train_file_handler.setFormatter(logging.Formatter('%(message)s'))
    train_log.addHandler(train_file_handler)

    train_log.info('Start Training\n')
    for epoch in range(epochs):
        train_log.info(f'Epoch {epoch}')

        train_start = time.time()
        model, train_loss = train_epoch(optimizer, model, loss_func, train_dataloader, gpu)
        train_end = time.time()
        train_time = train_end - train_start
        train_log.info(f'Training Time: {train_time:.3f} s')
        train_times.update(train_time)

        scheduler.step()

        valid_start = time.time()
        classification_dict, roc_auc, pr_auc, valid_loss = eval_epoch(
            model,
            loss_func,
            valid_dataloader,
            class_names,
            gpu
        )
        valid_end = time.time()
        valid_time = valid_end - valid_start
        train_log.info(f'Validation Time: {train_time:.3f} s')
        valid_times.update(valid_time)

        valid_f1 = classification_dict["macro avg"]["f1-score"]
        train_log.info(f'Train Loss: {train_loss.avg:.3f}')
        train_log.info(f'Valid Loss: {valid_loss.avg:.3f}')
        train_log.info(f'Valid Macro-F1: {valid_f1:.4f}')
        train_log.info(f'Valid ROC-AUC: {roc_auc:.4f}')
        train_log.info(f'Valid PR-AUC: {pr_auc:.4f}')
        train_log.info('\n')

        model_path = dir_path / 'model_epoch_{epoch}.pt'
        torch.save(model.state_dict(), model_path.resolve())

        if valid_loss.avg < min_valid_loss:
            min_valid_loss = valid_loss.avg
            torch.save(model.state_dict(), best_model_path.resolve())

    train_log.info(f'Average Train Time: {train_times.avg:.3f} s')
    train_log.info(f'Average Valid Time: {valid_times.avg:.3f} s')

    train_log.removeHandler(train_file_handler)
    train_file_handler.close()

    return model

def eval_model(
    model: object,
    loss_func: object,
    test_dataloader: object,
    true_matrix: torch.Tensor,
    gpu: torch.device,
    class_names: List[str],
    dir_path: str,
):
    """
    Evaluate the model with test dataloader.
    This function gives a more info compared to eval_epoch.
    """
    eval_log = logging.getLogger()
    eval_log.setLevel(logging.INFO)
    eval_file_handler = logging.FileHandler(str(dir_path / 'model_eval'), mode='w')
    eval_file_handler.setFormatter(logging.Formatter('%(message)s'))
    eval_log.addHandler(eval_file_handler)

    pbar = tqdm(test_dataloader)
    model.eval()
    with torch.no_grad():
        test_loss = AverageMeter()
        predictions = []
        target_probs = []
        answers = []

        if model.model_name == 'selor':
            _, n_data = true_matrix.shape
            confidences = AverageMeter()
            consistencies = AverageMeter()
            duplicates = AverageMeter()
            coverages = AverageMeter()
            uniques = AverageMeter()
            lengths = []

        eval_start = time.time()
        for batch in test_dataloader:
            inputs, y = batch
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
                    _, ind = torch.max(atom_prob, dim=-1)
                    ind_list.append(ind)

                # Calculate Confidence
                cp_list = torch.stack(cp_list, dim=1)
                cp_mean_list = torch.mean(cp_list, dim=1)
                max_cp, _ = torch.max(cp_mean_list, dim=-1)
                confidence = torch.abs(max_cp - 0.5) * 2

                # Calculate Consistency
                _, n_head, _ = cp_list.shape
                _, decision = torch.max(cp_list, dim=-1)
                consistency = (decision == decision[:, 0].unsqueeze(dim=-1).repeat(1, n_head)).int()
                consistency = (torch.sum(consistency, dim=-1) == n_head).float()

                ind_list = torch.stack(ind_list, dim=1)
                _, num_head, max_antecedent_len = ind_list.shape
                ind_list = ind_list.view(batch_size, num_head * max_antecedent_len)

                # Calculate the number of unique atoms
                unique = []
                for i in ind_list:
                    unique.append(len(set(i.tolist())))

                # Calculate the length of antecedent
                length = torch.sum((ind_list != 0), dim=1)

                # Calculate the number of duplicated atoms
                token, _ = torch.mode(ind_list, dim=1)
                check = ind_list==token.unsqueeze(dim=-1)
                duplicate = torch.sum(check, dim=-1)

                # Calculate the coverage of the antecedent
                mat_antecedent_prob = torch.stack(atom_prob_list, dim=1)
                cover_antecedent_prob = torch.sum(mat_antecedent_prob, dim=2)
                cover_antecedent_prob = torch.matmul(cover_antecedent_prob, true_matrix)
                mat_satis = (cover_antecedent_prob == max_antecedent_len)
                mat_satis = torch.sum(mat_satis.float(), dim=-1)
                mat_coverage = mat_satis / n_data
                coverage = torch.mean(mat_coverage, dim=-1)

                confidences.update(confidence.mean().item(), batch_size)
                consistencies.update(consistency.mean().item(), batch_size)
                uniques.update(torch.Tensor(unique).float().mean().item())
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

        pbar.close()
        eval_end = time.time()
        eval_time = eval_end - eval_start

        predictions = torch.stack(predictions).cpu().tolist()
        target_probs = torch.stack(target_probs).cpu().tolist()
        answers = torch.stack(answers).cpu().tolist()

        if model.model_name == 'selor':
            count_length = Counter(torch.stack(lengths).int().tolist())
            dist_length = {}
            for length, count in count_length.items():
                dist_length[length] = count / len(answers)

        eval_log.info(f'Avg Test Loss: {test_loss.avg:.4f}')
        eval_log.info(f'Evaluation Time: {eval_time:.3f} s')

        if model.model_name == 'selor':
            eval_log.info(f'Confidence: {confidences.avg:.4f}')
            eval_log.info(f'Consistency: {consistencies.avg:.4f}')
            eval_log.info(f'Duplicate: {duplicates.avg:.4f}')
            eval_log.info(f'Unique: {uniques.avg:.4f}')
            eval_log.info(f'Coverage: {coverages.avg:.4f}')
            eval_log.info('Length')
            for i in range(max_antecedent_len + 1):
                if i in dist_length:
                    eval_log.info(f'{i}: {dist_length[i]:.4f}')

        roc_auc = roc_auc_score(answers, target_probs)
        precision, recall, _ = precision_recall_curve(
            answers,
            predictions,
            pos_label=1
        )
        pr_auc = auc(recall, precision)

        eval_log.info('Prediction Performance:')
        c_report = classification_report(
            answers,
            predictions,
            target_names=class_names,
            digits=4
        )
        eval_log.info(f'{c_report}')
        eval_log.info(f'ROC-AUC: {roc_auc:.4f}')
        eval_log.info(f'PR-AUC: {pr_auc:.4f}')

        eval_log.removeHandler(eval_file_handler)
        eval_file_handler.close()

def get_explanation(
    model: object,
    true_matrix: torch.Tensor,
    atom_pool: object,
    inputs: Tuple[torch.Tensor, ...],
    class_names: List[str],
    gpu: torch.device,
) -> Tuple[Dict[str, float], List[str], List[float]]:
    """
    Get an explanation for the instance.
    """
    model.eval()

    with torch.no_grad():
        inputs = [i.to(gpu).unsqueeze(dim=0) for i in inputs]
        outputs, atom_prob_list, _ = model(
            inputs
        )
        outputs = outputs.squeeze(dim=0)
        outputs = torch.exp(outputs)

        antecedent_list = []
        coverage_list = []

        for atom_prob in atom_prob_list:
            _, ind = torch.max(atom_prob, dim=-1)

            ind = ind.squeeze(dim=0)
            antecedents = [atom_pool.atoms[atom_pool.atom_id2key[i]] for i in ind]
            antecedent_list.append(antecedents)

            cover_antecedent_prob = torch.sum(atom_prob, dim=1)
            cover_antecedent_prob = torch.matmul(cover_antecedent_prob, true_matrix)
            mat_satis = (cover_antecedent_prob == model.antecedent_len)
            mat_satis = torch.sum(mat_satis.float(), dim=-1)
            coverage = mat_satis / model.n_data
            coverage_list.append(round(coverage.item(), 6))

        class_probs = {}
        for i, _ in enumerate(outputs):
            class_probs[f'{class_names[i]}'] = round(outputs[i].item(), 4)

        antecedents = []
        for antecedent in antecedent_list:
            antecedent_str = ' & '.join([atom.display_str for atom in antecedent])
            antecedents.append(antecedent_str)

        return class_probs, antecedents, coverage_list

def get_all_explanation(
    model: object,
    dataset: str,
    test_df: pd.DataFrame,
    atom_pool: object,
    true_matrix: torch.Tensor,
    gpu: torch.device,
    tf_tokenizer: object,
    atom_tokenizer: object,
    class_names: List[str],
    tabular_info: Tuple[dict, dict, dict]=None,
    tabular_column_type: Tuple[list, list, list]=None,
) -> Tuple[List[str], List[dict]]:
    """
    Extract all explanations of given test dataset
    """
    if dataset == 'adult':
        categorical_x_col, numerical_x_col, y_col = tabular_column_type
        cat_map, _, numerical_max = tabular_info

    exp_list = []
    result_list = []
    for target_id in tqdm(range(len(test_df)), desc='Extracting Explanations'):
        exp = ''

        target_context = ''
        row = test_df.iloc[target_id,:]

        if dataset == 'yelp':
            target_context += f'text: {row["text"]}\n'
        elif dataset == 'clickbait':
            target_context += f'title: {row["title"]}\n'
            target_context += f'text: {row["text"]}\n'
        elif dataset == 'adult':
            for key in row.index:
                value = row[key]
                if key in numerical_x_col:
                    target_context += f'{key}: {round(value * numerical_max[key], 1)}\n'
                elif key in y_col:
                    continue
                else:
                    if value == 1:
                        context, target = key.split('_')
                        assert context in categorical_x_col
                        cur = cat_map[f'{context}_idx2key'][int(float(target))]
                        target_context += f'{context}: {cur}\n'

        exp += f'{target_context}\n'

        inputs = get_single_input(
            row,
            dataset,
            atom_pool,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
        )

        class_probs, antecedents, coverage_list = get_explanation(
            model,
            true_matrix,
            atom_pool,
            inputs,
            class_names,
            gpu
        )
        pred = max(class_probs, key=class_probs.get)

        assert len(y_col) == 1
        y = int(row[y_col[0]])

        label = class_names[y]

        exp += f'Label: {label}\n'
        exp += f'Prediction: {pred}\n\n'
        exp += 'Class Probability\n'
        for class_name, prob in class_probs.items():
            exp += f'{class_name}: {prob}\n'

        exp += '\n'
        for i, antecedent in enumerate(antecedents):
            coverage = coverage_list[i]

            exp += f'Explanation {i}: {antecedent}\n'
            exp += f'Coverage: {coverage:.6f}\n'
            exp += '\n'

        result_dict = {
            'Id': target_id,
            'Target': target_context,
            'Label': label,
            'Prediction': pred,
            'Explanation': antecedents,
            'Class Probability': class_probs,
            'Coverage': coverage_list
        }

        result_list.append(result_dict)
        exp_list.append(exp)

    return exp_list, result_list

def get_base_embedding(
    model: object,
    train_dataloader: object,
    gpu: torch.device,
) -> torch.Tensor:
    """
    Get embedding of given base model and train dataloader.
    """
    assert model.model_name == 'base'

    h_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            inputs, y = batch
            inputs = [i.to(gpu) for i in inputs]
            y = y.to(gpu)

            _, h, _ = model(
                inputs
            )

            h_list.append(h.cpu())
    embeddings = torch.cat(h_list, dim=0)
    return embeddings
