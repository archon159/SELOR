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
from sklearn.metrics import classification_report, roc_auc_score
from dataset import get_class_names

# The function for training cp predictor.
def pretrain(
    ce_model,
    pretrain_train_dataloader,
    atom_embedding,
    n_data,
    weight_mu,
    weight_sigma,
    weight_coverage,
    args
):
    gpu = torch.device(f'cuda:{args.gpu}')
    
    n_atom, hidden_dim = atom_embedding.shape

    optimizer = torch.optim.AdamW(ce_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    ce_model.train()
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        pbar = tqdm(range(len(pretrain_train_dataloader)))

        ce_losses = []
        mu_losses = []
        sigma_losses = []
        coverage_losses = []
        for d in pretrain_train_dataloader:
            x, mu, sigma, n = d
            #mu = torch.stack(((1 - mu), mu), dim=1)

            batch_size, rule_len = x.shape
            # We pad dummy if the rule length is smaller than max_rule_len
            x = F.pad(x, (0, args.max_rule_len - rule_len))

            mu = mu.to(gpu)
            sigma = sigma.to(gpu)
            coverage = (n / n_data).to(gpu)

            e = atom_embedding[x, :].detach()
            e = e.to(gpu)
            mu_, sigma_, coverage_ = ce_model(e)

            # L2 distance as loss
            if args.dataset == 'yelp':
                mu_loss = torch.mean(F.pairwise_distance(mu.unsqueeze(dim=-1), mu_.unsqueeze(dim=-1)))
                sigma_loss = torch.mean(F.pairwise_distance(sigma.unsqueeze(dim=-1), sigma_.unsqueeze(dim=-1)))
            else:
                mu_loss = torch.mean(F.pairwise_distance(mu, mu_))
                sigma_loss = torch.mean(F.pairwise_distance(sigma, sigma_))
            
            coverage_loss = torch.mean(F.pairwise_distance(coverage.unsqueeze(dim=-1), coverage_.unsqueeze(dim=-1)))

            loss = weight_mu * mu_loss + weight_sigma * sigma_loss + weight_coverage * coverage_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ce_losses.append(loss.item())
            mu_losses.append(mu_loss.item())
            coverage_losses.append(coverage_loss.item())

            avg_ce_loss = torch.mean(torch.tensor(ce_losses)).item()
            avg_mu_loss = torch.mean(torch.tensor(mu_losses)).item()
            avg_sigma_loss = torch.mean(torch.tensor(mu_losses)).item()
            avg_coverage_loss = torch.mean(torch.tensor(coverage_losses)).item()

            pbar.set_description(f'Train Loss: {avg_ce_loss:.6f} ({avg_mu_loss:.6f}, {avg_sigma_loss:.6f}, {avg_coverage_loss:.6f})')
            pbar.update(1)
        pbar.close()

    return ce_model

def eval_pretrain(
    ce_model,
    pretrain_test_dataloader,
    atom_embedding,
    n_data,
    class_names,
    args,
):
    gpu = torch.device(f'cuda:{args.gpu}')
    n_atom, hidden_dim = atom_embedding.shape
    ce_model.eval()
    
    mu_pred = []
    mu_answer = []
    sigma_pred = []
    sigma_answer = []
    coverage_pred = []
    coverage_answer = []
    
    for d in tqdm(pretrain_test_dataloader):
        x, mu, sigma, n = d
        coverage = n / n_data

        batch_size, rule_len = x.shape
        x = F.pad(x, (0, 4 - rule_len))

        with torch.no_grad():
            e = atom_embedding[x, :].detach()
            e = e.to(gpu)
            mu_, sigma_, coverage_ = ce_model(e)

            mu_pred.extend(mu_)
            mu_answer.extend(mu)
            sigma_pred.extend(sigma_)
            sigma_answer.extend(sigma)
            coverage_pred.extend(coverage_)
            coverage_answer.extend(coverage)

    if args.dataset == 'yelp':
        mu_pred = torch.tensor(mu_pred)
        mu_answer = torch.tensor(mu_answer)
        
        sigma_pred = torch.tensor(sigma_pred)
        sigma_answer = torch.tensor(sigma_answer)
    else:
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
    #_, q_mu_pred = torch.max(mu_pred)
    
    if args.dataset == 'yelp':
        q_mu_pred = mu_pred > 0.5
        q_mu_answer = (mu_answer > 0.5)
    else:
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
    args,
):
    gpu = torch.device(f'cuda:{args.gpu}')
    
    train_loss = []
    pbar = tqdm(train_dataloader)

    model.train()
    for d in train_dataloader:
        if args.dataset in ['yelp', 'clickbait']:
            input_ids, attention_mask, x_, y = d
            input_ids = input_ids.to(gpu).squeeze(dim=1)
            attention_mask = attention_mask.to(gpu).squeeze(dim=1)
            y = y.to(gpu).squeeze(dim=1)
        
            # Run the model
            if model.model_name == 'base':
                inputs = input_ids, attention_mask
                outputs, _ = model(
                    inputs
                )
            elif model.model_name == 'rule_gen':
                x_ = x_.to(gpu)
                inputs = input_ids, attention_mask, x_
                # Run the model
                outputs, _, _ = model(
                    inputs
                )
            else:
                assert(0)
                
        elif args.dataset in ['adult']:
            x, x_, y = d
            x = x.to(gpu)
            y = y.to(gpu).squeeze(dim=1)
            
            if model.model_name == 'base':
                inputs = x
                outputs, _ = model(
                    inputs
                )
            elif model.model_name == 'rule_gen':
                x_ = x_.to(gpu)
                inputs = x, x_
                outputs, _, _ = model(
                    inputs
                )
                
            else:
                assert(0)
        else:
            assert(0)

        loss = loss_func(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())

        pbar.update(1)
        average_train_loss = sum(train_loss)/len(train_loss)
        pbar.set_description(f'Train Loss: {average_train_loss:.3f}')
        # break
        
    pbar.close()

    avg_train_loss = sum(train_loss) / len(train_loss)

    return model, average_train_loss

def eval_epoch(
    model,
    loss_func,
    valid_dataloader,
    args,
):
    gpu = torch.device(f'cuda:{args.gpu}')
    
    pbar = tqdm(valid_dataloader)
    model.eval()
    with torch.no_grad():
        valid_loss = []
        predictions = []
        answers = []
        for d in valid_dataloader:
            if args.dataset in ['yelp', 'clickbait']:
                input_ids, attention_mask, x_, y = d
                input_ids = input_ids.to(gpu).squeeze(dim=1)
                attention_mask = attention_mask.to(gpu).squeeze(dim=1)
                y = y.to(gpu).squeeze(dim=1)

                # Run the model
                if model.model_name == 'base':
                    inputs = input_ids, attention_mask
                    outputs, _ = model(
                        inputs
                    )
                elif model.model_name == 'rule_gen':
                    x_ = x_.to(gpu)
                    inputs = input_ids, attention_mask, x_
                    # Run the model
                    outputs, _, _ = model(
                        inputs
                    )
                else:
                    assert(0)

            elif args.dataset in ['adult']:
                x, x_, y = d
                x = x.to(gpu)
                y = y.to(gpu).squeeze(dim=1)

                if model.model_name == 'base':
                    inputs = x
                    outputs, _ = model(
                        inputs
                    )
                elif model.model_name == 'rule_gen':
                    x_ = x_.to(gpu)
                    inputs = x, x_
                    outputs, _, _ = model(
                        inputs
                    )
                    
                else:
                    assert(0)
            else:
                assert(0)
                
            loss = loss_func(outputs, y) 
            valid_loss.append(loss.item())
            
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            answers.extend(y)
            
            pbar.update(1)
            average_valid_loss = sum(valid_loss)/len(valid_loss)
            pbar.set_description(f'Valid Loss: {average_valid_loss:.3f}')
            # break
            
            
        pbar.close()
        predictions = torch.stack(predictions).cpu().tolist()
        answers = torch.stack(answers).cpu().tolist()
        
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        
        class_names = get_class_names(args.dataset)
        classification_dict = classification_report(answers, predictions, target_names=class_names, output_dict=True)
        
        return classification_dict, average_valid_loss
    
def train(
    model,
    loss_func,
    train_dataloader,
    valid_dataloader,
    args,
    dir_path,
    flog,
):
    DIR_PATH = dir_path
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    USE_SCHEDULER = True
    VERBOSE = True

    if USE_SCHEDULER:
        scheduler = ExponentialLR(optimizer, gamma=0.95)

    print('Start Training', file=flog, flush=True)
    
    min_valid_loss = 100.0
    
    BEST_MODEL_PATH = f'{DIR_PATH}/model_best.pt'
    
    train_time_list = []
    valid_time_list = []
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}', file=flog, flush=True)
        
        train_start = time.time()
        model, train_loss = train_epoch(optimizer, model, loss_func, train_dataloader, args)
        train_end = time.time()
        train_time = train_end - train_start
        print(f'Training Time: {train_time:.3f} s', file=flog, flush=True)
        train_time_list.append(train_time)
        
        if USE_SCHEDULER:
            scheduler.step()
            
        valid_start = time.time()
        classification_dict, valid_loss = eval_epoch(model, loss_func, valid_dataloader, args)
        valid_end = time.time()
        valid_time = valid_end - valid_start
        print(f'Validation Time: {train_time:.3f} s', file=flog, flush=True)
        valid_time_list.append(valid_time)

        if VERBOSE:
            print(f'Train Loss: {train_loss:.3f}', file=flog, flush=True)
            print(f'Valid Loss: {valid_loss:.3f}', file=flog, flush=True)
            print('Valid Macro-F1:', classification_dict['macro avg']['f1-score'], file=flog, flush=True)
            print('\n', file=flog, flush=True)

        MODEL_PATH = f'{DIR_PATH}/model_epoch_{epoch}.pt'
        RESULT_PATH = f'{DIR_PATH}/model_validation_result_{epoch}.json'
        torch.save(model.state_dict(), MODEL_PATH)
        
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        
        valid_loss_dict = {}
        valid_loss_dict['epoch'] = epoch
        valid_loss_dict['valid_loss'] = valid_loss
        with open(RESULT_PATH, "w") as f:
            json.dump(valid_loss_dict, f)
            
    avg_train_time = sum(train_time_list) / len(train_time_list)
    avg_valid_time = sum(valid_time_list) / len(valid_time_list)
    print('\n')
    print(f'Average Training Time: {avg_train_time:.3f} s', file=flog, flush=True)
    print(f'Average Validation Time: {avg_valid_time:.3f} s', file=flog, flush=True)
    
def eval_model(
    model,
    loss_func,
    test_dataloader,
    args,
    feval,
    true_matrix=None,
    n_data=0,
):
    gpu = torch.device(f'cuda:{args.gpu}')
    
    pbar = tqdm(test_dataloader)
    model.eval()
    with torch.no_grad():
        test_loss = []
        predictions = []
        target_probs = []
        answers = []
        
        if model.model_name == 'rule_gen':
            confidences = []
            consistencies = []
            duplicates = []
            coverages = []
            uniques = []
            lengths = []
            
        eval_start = time.time()
        for d in test_dataloader:
            if args.dataset in ['yelp', 'clickbait']:
                input_ids, attention_mask, x_, y = d
                input_ids = input_ids.to(gpu).squeeze(dim=1)
                attention_mask = attention_mask.to(gpu).squeeze(dim=1)
                y = y.to(gpu).squeeze(dim=1)

                # Run the model
                if model.model_name == 'base':
                    inputs = input_ids, attention_mask
                    outputs, _ = model(
                        inputs
                    )
                elif model.model_name == 'rule_gen':
                    x_ = x_.to(gpu)
                    inputs = input_ids, attention_mask, x_
                    # Run the model
                    outputs, atom_prob_list, cp_list = model(
                        inputs
                    )
                else:
                    assert(0)

            elif args.dataset in ['adult']:
                x, x_, y = d
                x = x.to(gpu)
                y = y.to(gpu).squeeze(dim=1)

                if model.model_name == 'base':
                    inputs = x
                    outputs, _ = model(
                        inputs
                    )
                elif model.model_name == 'rule_gen':
                    x_ = x_.to(gpu)
                    inputs = x, x_
                    outputs, atom_prob_list, cp_list = model(
                        inputs
                    )
                    
                else:
                    assert(0)
            else:
                assert(0)
            
            if model.model_name == 'rule_gen':
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
                confidences.extend(confidence)

                batch_size, num_head, n_class = cp_list.shape

                # Calculate Consistency
                _, decision = torch.max(cp_list, dim=-1)
                cs = (decision == decision[:, 0].unsqueeze(dim=-1).repeat(1, num_head)).int()
                consistency = (torch.sum(cs, dim=-1) == num_head).float()
                consistencies.extend(consistency)

                ind_list = torch.stack(ind_list, dim=1)
                batch_size, num_head, rule_length = ind_list.shape
                ind_list = ind_list.view(batch_size, num_head * rule_length)

                # Calculate the number of unique atoms
                unique = []
                for i in ind_list:
                    unique.append(len(set(i.tolist())))
                uniques.extend(torch.tensor(unique).float())

                # Calculate the length of rule
                length = torch.sum((ind_list != 0), dim=1)
                lengths.extend(length.float())

                # Calculate the number of duplicated atoms
                token, _ = torch.mode(ind_list, dim=1)
                check = ind_list==token.unsqueeze(dim=-1)
                duplicate = torch.sum(check, dim=-1)
                duplicates.extend(duplicate.float())

                # Calculate the coverage of the rule
                mat_rule_prob = torch.stack(atom_prob_list, dim=1)
                cover_rule_prob = torch.sum(mat_rule_prob, dim=2)
                cover_rule_prob = torch.matmul(cover_rule_prob, true_matrix)
                mat_satis = (cover_rule_prob == rule_length)
                mat_satis = torch.sum(mat_satis.float(), dim=-1)
                mat_coverage = mat_satis / n_data
                coverage = torch.mean(mat_coverage, dim=-1)
                coverages.extend(coverage.float())
            
            loss = loss_func(outputs,y)
            test_loss.append(loss.item())
            
            _, preds = torch.max(outputs, dim=1)
            target_prob = torch.exp(outputs)[:, 1]
            target_probs.extend(target_prob)
            
            predictions.extend(preds)
            answers.extend(y)
            
            pbar.update(1)
            average_test_loss = sum(test_loss)/len(test_loss)
            pbar.set_description(f'Test Loss: {average_test_loss:.3f}')
            # break
            
        pbar.close()
        eval_end = time.time()
        eval_time = eval_end - eval_start
        
        avg_test_loss = sum(test_loss) / len(test_loss)
        
        predictions = torch.stack(predictions).cpu().tolist()
        target_probs = torch.stack(target_probs).cpu().tolist()
        answers = torch.stack(answers).cpu().tolist()
        
        if model.model_name == 'rule_gen':
            avg_confidence = torch.mean(torch.stack(confidences)).item()
            avg_consistency = torch.mean(torch.stack(consistencies)).item()
            avg_duplicate = torch.mean(torch.stack(duplicates)).item()
            avg_unique = torch.mean(torch.stack(uniques)).item()
            avg_coverage = torch.mean(torch.stack(coverages)).item()
            count_length = Counter(torch.stack(lengths).int().tolist())
            dist_length = {}
            for k, v in count_length.items():
                dist_length[k] = v / len(answers)

        print(f'Avg Test Loss: {avg_test_loss:.4f}', file=feval, flush=True)
        print(f'Evaluation Time: {eval_time:.3f} s', file=feval, flush=True)
        
        if model.model_name == 'rule_gen':
            print(f'Confidence: {avg_confidence:.4f}', file=feval, flush=True)
            print(f'Consistency: {avg_consistency:.4f}', file=feval, flush=True)
            print(f'Duplicate: {avg_duplicate:.4f}', file=feval, flush=True)
            print(f'Unique: {avg_unique:.4f}', file=feval, flush=True)
            print(f'Coverage: {avg_coverage:.6f}', file=feval, flush=True)
            print(f'Length ', file=feval, flush=True)
            for i in range(rule_length + 1):
                if i in dist_length:
                    print(f'{i}: {dist_length[i]:.4f}', file=feval, flush=True)
        
        print('Prediction Performance:', file=feval, flush=True)
        class_names = [str(c) for c in get_class_names(args.dataset)]
        print(classification_report(answers, predictions, target_names=class_names, digits=4), file=feval, flush=True)
        auc = roc_auc_score(answers, target_probs)
        print(f'AUC: {auc:.4f}', file=feval, flush=True)
        print('\n', file=feval, flush=True)
        
def get_train_embedding(
    model,
    train_dataloader,
    args,
):
    gpu = torch.device(f'cuda:{args.gpu}')
    emb_list = []
    
    model.eval()
    with torch.no_grad():
        for d in tqdm(train_dataloader):
            if args.dataset in ['yelp', 'clickbait']:
                input_ids, attention_mask, x_, y = d
                input_ids = input_ids.to(gpu).squeeze(dim=1)
                attention_mask = attention_mask.to(gpu).squeeze(dim=1)
                y = y.to(gpu).squeeze(dim=1)
                
                inputs = input_ids, attention_mask
                
            elif args.dataset in ['adult']:
                x, x_, y = d
                x = x.to(gpu)
                y = y.to(gpu).squeeze(dim=1)

                inputs = x

            # Run the model
            assert(model.model_name == 'base')
            _, emb = model(
                inputs
            )

            emb_list.append(emb.cpu())
    embeddings = torch.cat(emb_list, dim=0)
    return embeddings