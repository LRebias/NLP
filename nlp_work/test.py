import torch
import torch.nn as nn
from main import cal_performance, load_model_and_tokenizer
from sklearn.metrics import classification_report, confusion_matrix
from data.process import split_and_load_dataset, process_test, make_batch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

def tst(model, tokenizer, idx2label, criterion, test_loader, device, with_label=True):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    result = []
    preds = []
    Labels = [] if with_label else None
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = make_batch(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask)
            if with_label:
                loss = criterion(outputs.logits, labels)
            logits = outputs.logits

            pred = torch.argmax(logits, dim=-1).tolist()
            for i in range(len(input_ids)):
                cls = idx2label[pred[i]]
                preds.append(cls)
                if with_label:
                    Labels.append(idx2label[labels[i].item()])


            logits = logits.detach().cpu().numpy()

    if with_label:
        avg_val_accuracy = total_eval_accuracy / len(test_loader)
        print("Accuracy: %.4f" % (avg_val_accuracy))
        print("Average test loss: %.4f" % (total_eval_loss / len(test_loader)))
        print("-------------------------------")
    return result, preds, Labels

def gen_res(preds):
    index = list(range(1, len(preds) + 1))
    d = {'ID': index, 'Last Label': preds}
    df = pd.DataFrame(d)
    df.to_csv('./res.csv', sep=',', columns=['ID', 'Last Label'], index=False, header=True)
    print('has save to {}'.format('./res.csv'))


def get_preds_and_labels(model, test_loader, device, with_label=True):
    model.eval()
    preds = []
    Labels = [] if with_label else None
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = make_batch(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = F.softmax(logits, dim=-1)
            preds.append(batch_preds)
            if with_label:
                Labels.extend(labels.tolist())
    total_preds = torch.cat(preds)
    return total_preds, Labels

if __name__ == '__main__':
    with_label = False
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    _, n_labels, cnt, idx2label = process_test('./train_data.csv')
    data = process_test('./test_data_new.csv', with_label=with_label)
    criterion = nn.CrossEntropyLoss()

    lm_weight = 2.8#2.7
    

    model_paths = ['./checkpoints/bert/222']
    n_models = len(model_paths)
        
    models_preds = []
    for model_path in tqdm(model_paths):

        model, tokenizer = load_model_and_tokenizer(model_path, device)
        *_, test_loader = split_and_load_dataset(data, tokenizer, max_len=64, batch_size=32, with_label=with_label, test_size=1.0, shuf=False)
        single_model_preds, Labels = get_preds_and_labels(model, test_loader, device, with_label=with_label)
        models_preds.append(single_model_preds)
        del model, tokenizer, test_loader
            
    if lm_weight != 0:
        print('combine with language model.')
        lm_preds = get_lm_preds(alpha=lm_weight, device=device)
        models_preds.append(lm_preds)
        n_models += lm_weight
        
    avg_preds = sum(models_preds) / n_models

    preds = torch.argmax(avg_preds, dim=-1).tolist()

    real_preds = list(map(lambda x: idx2label[x], preds))
    gen_res(real_preds)