#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import argparse
import csv
import logging
import os
import random
import pickle
import sys

import numpy as np
import wandb 

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, BCEWithLogitsLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    BertForNextSentencePrediction,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from models_early_v_cls_ablation import *
from transformers.optimization import AdamW

import sys
sys.path.append("..")
from models.subNets.BertTextEncoder import BertTextEncoderRegressionHead
from global_config import *


def return_unk():
    return 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, choices=["HKT","language_only", "acoustic_only", "visual_only","hcf_only"], default="HKT",
)

parser.add_argument("--dataset", type=str, choices=["humor", "sarcasm",  "mosi"], default="sarcasm")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=85)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--cross_n_layers", type=int, default=1)
parser.add_argument("--cross_n_heads", type=int, default=4)
parser.add_argument("--fusion_dim", type=int, default=172)
parser.add_argument("--dropout", type=float, default=0.2366)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--include_v", type=str, choices=["y","n"], default = "y")
parser.add_argument("--include_t", type=str, choices=["y","n"], default = "y")
parser.add_argument("--include_a", type=str, choices=["y","n"], default = "y")
parser.add_argument("--include_h", type=str, choices=["y","n"], default = "y")
parser.add_argument("--save_preds", type=str, choices=["True","False"], default="False")

parser.add_argument("--seed", type=int, default=100)

parser.add_argument("--learning_rate", type=float, default=0.000005)
parser.add_argument("--learning_rate_t", type=float, default=0.00003)
parser.add_argument("--learning_rate_a", type=float, default=0.003)
parser.add_argument("--learning_rate_h", type=float, default=0.0003)
parser.add_argument("--learning_rate_v", type=float, default=0.003)
parser.add_argument("--warmup_ratio", type=float, default=0.07178)
parser.add_argument("--save_weight", type=str, choices=["True","False"], default="False")


args = parser.parse_args()



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, visual, acoustic,hcf,label_id,data_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.visual = visual
        self.acoustic = acoustic
        self.hcf = hcf
        self.label_id = label_id
        self.data_id = data_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    pop_count = 0
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) == 0:
            tokens_b.pop()
        else:
            pop_count += 1
            tokens_a.pop(0)
    return pop_count

#albert tokenizer split words in to subwords. "_" marker helps to find thos sub words
#our acoustic and visual features are aligned on word level. So we just create copy the same 
#visual/acoustic vectors that belong to same word.
def get_inversion(tokens, SPIECE_MARKER="▁"):
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)
    return inversions


def convert_humor_to_features(examples, tokenizer, punchline_only=False):
    features = []

    for (ex_index, example) in enumerate(examples):
        
        #p denotes punchline, c deontes context
        #hid is the utterance unique id. these id's are provided by the authors of urfunny and mustard
        #label is either 1/0 . 1=humor, 0=not humor
        (
            (p_words, p_visual, p_acoustic, p_hcf),
            (c_words, c_visual, c_acoustic, c_hcf),
            hid,
            label
        ) = example
                
        text_a = ". ".join(c_words)
        text_b = p_words + "."
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        
        inversions_a = get_inversion(tokens_a)
        inversions_b = get_inversion(tokens_b)

        pop_count = _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)

        inversions_a = inversions_a[pop_count:]
        inversions_b = inversions_b[: len(tokens_b)]

        visual_a = []
        acoustic_a = []
        hcf_a=[]        
        #our acoustic and visual features are aligned on word level. So we just 
        #create copy of the same visual/acoustic vectors that belong to same word.
        #because ber tokenizer split word into subwords
        for inv_id in inversions_a:
            visual_a.append(c_visual[inv_id, :])
            acoustic_a.append(c_acoustic[inv_id, :])
            hcf_a.append(c_hcf[inv_id, :])
            


        visual_a = np.array(visual_a)
        acoustic_a = np.array(acoustic_a)
        hcf_a = np.array(hcf_a)
        
        visual_b = []
        acoustic_b = []
        hcf_b = []
        for inv_id in inversions_b:
            visual_b.append(p_visual[inv_id, :])
            acoustic_b.append(p_acoustic[inv_id, :])
            hcf_b.append(p_hcf[inv_id, :])
        
        visual_b = np.array(visual_b)
        acoustic_b = np.array(acoustic_b)
        hcf_b = np.array(hcf_b)
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

        acoustic_zero = np.zeros((1, ACOUSTIC_DIM_ALL))
        if len(tokens_a) == 0:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_zero, acoustic_b, acoustic_zero)
            )
        else:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_a, acoustic_zero, acoustic_b, acoustic_zero)
            )

        visual_zero = np.zeros((1, VISUAL_DIM_ALL))
        if len(tokens_a) == 0:
            visual = np.concatenate((visual_zero, visual_zero, visual_b, visual_zero))
        else:
            visual = np.concatenate(
                (visual_zero, visual_a, visual_zero, visual_b, visual_zero)
            )
        
        
        hcf_zero = np.zeros((1,4))
        if len(tokens_a) == 0:
            hcf = np.concatenate((hcf_zero, hcf_zero, hcf_b, hcf_zero))
        else:
            hcf = np.concatenate(
                (hcf_zero, hcf_a, hcf_zero, hcf_b, hcf_zero)
                
            )
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(input_ids)
            
        acoustic_padding = np.zeros(
            (args.max_seq_length - len(input_ids), acoustic.shape[1])
        )
        acoustic = np.concatenate((acoustic, acoustic_padding))
        #original urfunny acoustic feature dimension is 81.
        #we found many features are highly correllated. so we removed
        #highly correlated feature to reduce dimension
        acoustic=np.take(acoustic, acoustic_features_list,axis=1)
        
        visual_padding = np.zeros(
            (args.max_seq_length - len(input_ids), visual.shape[1])
        )
        visual = np.concatenate((visual, visual_padding))
        #original urfunny visual feature dimension is more than 300.
        #we only considred the action unit and face shape parameter features
        visual = np.take(visual, visual_features_list,axis=1)
        
        
        hcf_padding= np.zeros(
            (args.max_seq_length - len(input_ids), hcf.shape[1])
        )
        
        hcf = np.concatenate((hcf, hcf_padding))
        
        padding = [0] * (args.max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length
        assert hcf.shape[0] == args.max_seq_length
        
        label_id = float(label)
        
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                hcf=hcf,
                label_id=label_id,
                data_id=int(hid)
            )
        )
            
    return features



def convert_to_features_mosi(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        # if args.model == "bert-base-uncased":
        prepare_input = prepare_bert_input
        # elif args.model == "xlnet-base-cased":
        #     prepare_input = prepare_xlnet_input
        hcf = np.zeros((args.max_seq_length, HCF_DIM_ALL))

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Reduce dimensionality of the acoustic and visual features
        acoustic = np.take(acoustic, acoustic_features_list, axis=1)
        visual = np.take(visual, visual_features_list, axis=1)

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                hcf = hcf,
                label_id=label_id,
                data_id = ex_index
            )
        )
    return features


def prepare_bert_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM_ALL))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM_ALL))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM_ALL))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM_ALL))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    # if model == "bert-base-uncased":
    return BertTokenizer.from_pretrained("bert-base-uncased")
    # elif model == "xlnet-base-cased":
    #     return XLNetTokenizer.from_pretrained(model)
    # else:
    #     raise ValueError(
    #         "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
    #             model
    #         )
    #     )


def get_appropriate_dataset(data, tokenizer, parition, mosi=False):
    
    if mosi:
        features = convert_to_features_mosi(data,tokenizer=tokenizer,max_seq_length=args.max_seq_length)
    else:
        features = convert_humor_to_features(data, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    hcf = torch.tensor([f.hcf for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_data_ids = torch.tensor([f.data_id for f in features], dtype=torch.float)
    

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        hcf,
        all_label_ids,
        all_data_ids    )
    
    return dataset


def set_up_data_loader():
    if args.dataset=="humor":
        data_file = "ur_funny.pkl"
    elif args.dataset=="sarcasm":
        data_file = "mustard.pkl"
    elif args.dataset=="mosi":
        mosi=True
        data_file = 'mosi.pkl'

    with open(
        os.path.join(DATASET_LOCATION, data_file),
        "rb",
    ) as handle:
        all_data = pickle.load(handle)
        
    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    if mosi:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")


    train_dataset = get_appropriate_dataset(train_data, tokenizer, "train",mosi)
    dev_dataset = get_appropriate_dataset(dev_data, tokenizer, "dev",mosi)
    test_dataset = get_appropriate_dataset(test_data, tokenizer, "test",mosi)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    
    
    return train_dataloader, dev_dataloader, test_dataloader

def train_epoch(model, train_dataloader, optimizer, scheduler, loss_fct):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

        batch = tuple(t.to(DEVICE) for t in batch)
        (
            input_ids,
            visual,
            acoustic,
            input_mask,
            segment_ids,
            hcf,
            label_ids,
            data_ids
        ) = batch
        
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        if args.model == "language_only":
            outputs = model(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
        elif args.model == "acoustic_only":
            outputs = model(
                acoustic
            )
        elif args.model == "visual_only":
            outputs = model(
                visual
            )
        elif args.model=="hcf_only":
            outputs=model(hcf)
            
        elif args.model=="HKT":
            outputs = model(input_ids, visual, acoustic, hcf, attention_mask=input_mask, token_type_ids=segment_ids)
                    
        
            
        logits = outputs[0]
        
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        loss.backward()
        
        for o_i in range(len(optimizer)):
            optimizer[o_i].step()
            scheduler[o_i].step()
        
        model.zero_grad()

    return tr_loss/nb_tr_steps



def eval_epoch(model, dev_dataloader, loss_fct):
    
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            (
                input_ids,
                visual,
                acoustic,
                input_mask,
                segment_ids,
                hcf,
                label_ids,
                data_ids

            ) = batch
                    
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
    
            if args.model == "language_only":
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            elif args.model == "acoustic_only":
                outputs = model(
                    acoustic
                )
            elif args.model == "visual_only":
                outputs = model(
                    visual
                )
            elif args.model=="hcf_only":
                outputs=model(hcf)
                
            elif args.model=="HKT":
                outputs = model(input_ids, visual, acoustic,hcf, token_type_ids=segment_ids, attention_mask=input_mask,)
            
            
            logits = outputs[0]
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
    
            dev_loss += loss.item()
            nb_dev_examples += input_ids.size(0)
            nb_dev_steps += 1

    return dev_loss/nb_dev_steps

def test_epoch(model, test_data_loader, loss_fct, regression = False, save_features = True):
    """ Epoch operation in evaluation phase """
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    all_labels = []
    all_features = []
    all_ids = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data_loader, desc="Iteration")):
            
            batch = tuple(t.to(DEVICE) for t in batch)

            (
                input_ids,
                visual,
                acoustic,
                input_mask,
                segment_ids,
                hcf,
                label_ids,
                data_ids
            ) = batch
                    
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            
            if args.model == "language_only":
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            elif args.model == "acoustic_only":
                outputs = model(
                    acoustic
                )
            elif args.model == "visual_only":
                outputs = model(
                    visual
                )
            elif args.model=="hcf_only":
                outputs=model(hcf)
                
            elif args.model=="HKT":
                outputs = model(input_ids, visual, acoustic,hcf, token_type_ids=segment_ids, attention_mask=input_mask,)
            
            
            logits = outputs[0]


            if save_features:
                all_features.append(outputs[1].detach().cpu().numpy())
            
            
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            
            if not regression:
                print("NOT REG")
                logits = torch.sigmoid(logits)
            
            if len(preds) == 0:
                preds=logits.detach().cpu().numpy()
                all_labels=label_ids.detach().cpu().numpy()
                all_ids = data_ids.detach().cpu().numpy()
            else:
                preds= np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_labels = np.append(
                    all_labels, label_ids.detach().cpu().numpy(), axis=0
                )
                all_ids = np.append(all_ids,data_ids.detach().cpu().numpy(), axis=0)
                
                
        all_features = np.concatenate(all_features, axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.squeeze(preds)
        all_labels = np.squeeze(all_labels)
        all_ids = np.squeeze(all_ids)
        all_features = np.squeeze(all_features)

    if save_features:
        return preds, all_labels, eval_loss,all_ids, all_features
    else:
        return preds, all_labels, eval_loss,all_ids



def test_score_model(model, test_data_loader, loss_fct, exclude_zero=False,save_features=True):

    if save_features:
        predictions, y_test, test_loss,data_ids, features = test_epoch(model, test_data_loader, loss_fct, save_features=True)
        # Save features to disk or do further processing

    else:
        predictions, y_test, test_loss,data_ids = test_epoch(model, test_data_loader, loss_fct)
    
    predictions = predictions.round()

    f_score = f1_score(y_test, predictions, average="weighted")
    accuracy = accuracy_score(y_test, predictions)

		
    # Confusion Matrix
    data = zip(data_ids,predictions,y_test)
    performanceDict = dict([(str(x), (y, z)) for x, y, z in data])

    featureDict = dict([(str(x),y) for x,y in zip(data_ids,features)])

    # Classification Report
    cr = classification_report(y_test, predictions, target_names=['class_0', 'class_1'])
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    print("Accuracy:", accuracy,"F score:", f_score)
    return accuracy, f_score, test_loss,performanceDict,cr,conf_matrix, featureDict



def test_score_model_reg(model, test_data_loader, loss_fct, exclude_zero=False, save_features = True, regression=False, use_zero=False):

    if save_features:
        predictions, y_test, test_loss, data_ids, all_features = test_epoch(model, test_data_loader, loss_fct,regression, save_features=True )
        # Save features to disk or do further processing
        featureList = [x for x in zip(y_test,all_features)]


        data = zip(data_ids,predictions,y_test)
        performanceDict = dict([(str(x), (y, z)) for x, y, z in data])

        #MAE includes neutral
        mae = np.mean(np.absolute(predictions - y_test))
        corr = np.corrcoef(predictions, y_test)[0][1]


        #non neg vs neg class split
        predictNN = predictions >= 0
        y_testNN = y_test >= 0

        test_preds_a7 = np.clip(predictions, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
        mult_a7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(test_truth_a7))

        f_score_nn = f1_score(y_testNN, predictNN, average="weighted")
        accNN = accuracy_score(y_testNN, predictNN)

        # Classification Report
        cr1 = classification_report(y_testNN, predictNN, target_names=['class_0', 'class_1'])
        
        # Confusion Matrix
        conf_matrix1 = confusion_matrix(y_testNN, predictNN)

        #pos vs neg class split
        non_zeros = np.array(
            [i for i, e in enumerate(y_test) if e != 0 or use_zero])

        predictions = predictions[non_zeros]
        y_test = y_test[non_zeros]

        predict2 = predictions >= 0
        y_test2 = y_test >= 0

        f_score2 = f1_score(y_test2, predict2, average="weighted")
        acc2 = accuracy_score(y_test2, predict2)

        # Classification Report
        cr2 = classification_report(y_test2, predict2, target_names=['class_0', 'class_1'])
        
        # Confusion Matrix
        conf_matrix2 = confusion_matrix(y_test2, predict2)


        print("Mean Absolute Error:", mae, " AccNN: ", accNN, " Acc2 ", acc2, " Acc7 ", mult_a7,  " cor: ",corr," f_scoreNN: ",f_score_nn, " f_score2 ", f_score2)
        return accNN, acc2, mult_a7, mae, corr, f_score_nn, f_score2, test_loss, featureList,cr1,cr2,conf_matrix1,conf_matrix2,performanceDict

    else:
        predictions, y_test, test_loss,data_ids = test_epoch(model, test_data_loader, loss_fct, regression,save_features=False)
    


        data = zip(data_ids,predictions,y_test)
        performanceDict = dict([(str(x), (int(y), int(z))) for x, y, z in data])

        #MAE includes neutral
        mae = np.mean(np.absolute(predictions - y_test))
        corr = np.corrcoef(predictions, y_test)[0][1]


        #non neg vs neg class split
        predictNN = predictions >= 0
        y_testNN = y_test >= 0

        test_preds_a7 = np.clip(predictions, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
        mult_a7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(test_truth_a7))

        f_score_nn = f1_score(y_testNN, predictNN, average="weighted")
        accNN = accuracy_score(y_testNN, predictNN)

        # Classification Report
        cr1 = classification_report(y_test, predictions, target_names=['class_0', 'class_1'])
        
        # Confusion Matrix
        conf_matrix1 = confusion_matrix(y_test, predictions)

        #pos vs neg class split
        non_zeros = np.array(
            [i for i, e in enumerate(y_test) if e != 0 or use_zero])

        predictions = predictions[non_zeros]
        y_test = y_test[non_zeros]

        predict2 = predictions >= 0
        y_test2 = y_test >= 0

        f_score2 = f1_score(y_testNN, predictNN, average="weighted")
        acc2 = accuracy_score(y_testNN, predictNN)

        # Classification Report
        cr2 = classification_report(y_test2, predict2, target_names=['class_0', 'class_1'])
        
        # Confusion Matrix
        conf_matrix2 = confusion_matrix(y_test2, predict2)


        print("Mean Absolute Error:", mae, " AccNN: ", accNN, " Acc2 ", acc2, " Acc7 ", mult_a7,  " cor: ",corr," f_scoreNN: ",f_score_nn, " f_score2 ", f_score2)
        return accNN, acc2, mult_a7, mae, corr, f_score_nn, f_score2, test_loss,cr1,cr2,conf_matrix1,conf_matrix2,performanceDict

import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # Handle numpy arrays
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    loss_fct,
        regression=False,
):
       
    best_valid_test_accuracy = 0
    best_valid_test_fscore = 0
    best_valid_corr = 0
    best_valid_loss = 9e+9
    best_test_mae = 9e+9
    best_valid_corr = 0
    best_test_acc2a = 0
    best_test_acc2b = 0
    best_test_acc7 = 0
    best_valid_test_fscore_a = 0
    best_valid_test_fscore_b = 0
    run_name = str(wandb.run.id)
    valid_losses = []


    
    n_epochs=args.epochs
    patience = 5  # Define your patience value here
    epochs_without_improvement = 0
        
    
    for epoch_i in range(n_epochs):
        
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, loss_fct
        )
        valid_loss = eval_epoch(model, dev_dataloader, loss_fct)

        valid_losses.append(valid_loss)
        print(
            "\nepoch:{},train_loss:{}, valid_loss:{}".format(
                epoch_i, train_loss, valid_loss
            )
        )

        if regression==True:
            accNN, acc2, mult_a7, mae, corr, f_score_nn, f_score2, test_loss, featureList,cr1,cr2,conf_matrix1,conf_matrix2,performanceDict = test_score_model_reg(
                model, test_dataloader, loss_fct, regression=regression
            )

                
            if(best_test_mae >= mae):
                print(cr1)
                print(conf_matrix1)
                print(cr2)
                print(conf_matrix2)
                best_valid_loss = valid_loss
                best_test_mae = mae
                best_valid_corr = corr
                best_test_acc2a = acc2
                best_test_acc2b = accNN
                best_test_acc7 = mult_a7
                best_valid_test_fscore_a = f_score2
                best_valid_test_fscore_b = f_score_nn

                if(args.save_preds == "True"):
                    with open(f'performanceDict{wandb.run.id}.json', 'w') as fp:
                        import json
                        json.dump(performanceDict, fp, cls=NumpyEncoder)
                

                
                
                if(args.save_weight == "True"):
                    torch.save(model.state_dict(),'./best_weights/'+run_name+'.pt')

                with open(f"test_features_{wandb.run.id}.pkl", 'wb') as f:
                    pickle.dump(featureList, f)        

                #we report test_accuracy of the best valid loss (best_valid_test_accuracy)
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "test_loss": test_loss,
                        "best_valid_loss": best_valid_loss,
                        "best_test_mae": best_test_mae,
                        "best_test_acc2a": best_test_acc2a,
                        "best_test_acc2b": best_test_acc2b,
                        "best_test_acc7": best_test_acc7,
                        "best_valid_test_fscore_a": best_valid_test_fscore_a,
                        "best_valid_test_fscore_b": best_valid_test_fscore_b,
                        "best_valid_test_corr": best_valid_corr

                    }
                )

        else:
            test_accuracy, test_f_score, test_loss, predDict, classification_report, confusion_matrix,featureDict = test_score_model(
                model, test_dataloader, loss_fct
            )
            
                
            if(test_accuracy > best_valid_test_accuracy):
                best_valid_loss = valid_loss
                best_valid_test_accuracy = test_accuracy
                best_valid_test_fscore= test_f_score
                epochs_without_improvement = 0
                
                if(args.save_weight == "True"):
                    torch.save(model.state_dict(),'./best_weights/'+run_name+'.pt')
                
                if(args.save_preds == "True"):
                    with open('performanceDictX.json', 'w') as fp:
                        import json
                        json.dump(predDict, fp)
                with open(f"test_features_intermediate_{str(wandb.run.id)}.pkl", 'wb') as f:
                    pickle.dump(featureDict, f)
                # np.save(f"test_features_intermediate_{str(wandb.run.id)}.npy", all_features)
                # print(f"Size of features {all_features.shape}")

                
                print(classification_report)
                print(confusion_matrix)
        # else:
            # epochs_without_improvement +=1
            
        # If epochs without improvement exceeds patience, stop training
        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break
        
        #we report test_accuracy of the best valid loss (best_valid_test_accuracy)
        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "best_valid_loss": best_valid_loss,
                "best_valid_test_accuracy": best_valid_test_accuracy,
                "best_valid_test_fscore":best_valid_test_fscore
            }
        )
        



def get_optimizer_scheduler(params,num_training_steps,learning_rate=1e-5):
    
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )
    
    return optimizer,scheduler

def prep_for_training(num_training_steps):
    
    
    if args.model == "language_only":
        model = AlbertForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=1
        )
    elif args.model == "acoustic_only":
        model = Transformer(ACOUSTIC_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)
        
    elif args.model == "visual_only":
        model = Transformer(VISUAL_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)
        
    elif args.model=="hcf_only":
        model=Transformer(HCF_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)
        
    elif args.model == "HKT" :
        #HKT model has 4 unimodal encoders. But the language one is ALBERT pretrained model. But other enocders are
        #trained from scratch with low level features. We have found that many times most of the the gardients flows to albert encoders only as it
        #already has rich contextual representation. So in the beginning the gradient flows ignores other encoders which are trained from low level features. 
        # We found that if we intitalize the weights of the acoustic, visual and hcf encoders of HKT model from the best unimodal models that we already ran for ablation study then
        #the model converege faster. Other wise it takes very long time to converge. 
        if args.dataset=="humor":
            visual_model = Transformer(VISUAL_DIM, num_layers=7, nhead=3, dim_feedforward= 128)
            visual_model.load_state_dict(torch.load("./model_weights/init/humor/humorVisualTransformer.pt"), strict=False)
            acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=8, nhead=3, dim_feedforward = 256)
            acoustic_model.load_state_dict(torch.load("./model_weights/init/humor/humorAcousticTransformer.pt"), strict=False)
            hcf_model = Transformer(HCF_DIM, num_layers=3, nhead=2, dim_feedforward = 128)
            hcf_model.load_state_dict(torch.load("./model_weights/init/humor/humorHCFTransformer.pt"), strict=False)
            
        elif args.dataset=="sarcasm":
            visual_model = Transformer(VISUAL_DIM, num_layers=8, nhead=4, dim_feedforward=1024)
            visual_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmVisualTransformer.pt"), strict=False)
            acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=1, nhead=3, dim_feedforward=512)
            acoustic_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmAcousticTransformer.pt"), strict=False)
            hcf_model = Transformer(HCF_DIM, num_layers=8, nhead=4, dim_feedforward=128)
            hcf_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmHCFTransformer.pt"), strict=False)


        elif args.dataset=="mosi":
            visual_model = Transformer(VISUAL_DIM, num_layers=9, nhead=1, dim_feedforward=400)
            visual_model.load_state_dict(torch.load("./model_weights/init/mosi/mosiVisualTransformer1H9L400FC.pt"))
            acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=10, nhead=1, dim_feedforward=200)
            acoustic_model.load_state_dict(torch.load("./model_weights/init/mosi/mosiAcousticTransformer1H10L200FC.pt"))
            hcf_model = Transformer(HCF_DIM, num_layers=8, nhead=4, dim_feedforward=128) #not used 
            hcf_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmHCFTransformer.pt")) #not used

        if args.dataset == "mosi":
            text_model = BertTextEncoderRegressionHead(language='en', use_finetune=True)
        else:
            text_model = AlbertModel.from_pretrained('albert-base-v2')

        if args.include_v=="n":
            if args.dataset=="mosi":
                model = HKT_regression_no_V(text_model, visual_model, acoustic_model, args)
            else:
                model = HKT_no_V(text_model, visual_model, acoustic_model,hcf_model, args)

        elif args.include_a=="n":
            if args.dataset=="mosi":
                model = HKT_regression_no_A(text_model, visual_model, acoustic_model, args)
            else:
                model = HKT_no_A(text_model, visual_model, acoustic_model,hcf_model, args)

        elif args.include_t=="n":
            model = HKT_no_T(text_model, visual_model, acoustic_model,hcf_model, args)

        elif args.include_h=="n":
            model = HKT_no_H(text_model, visual_model, acoustic_model,hcf_model, args)
        elif args.dataset=="mosi":
            model = HKT_regression(text_model, visual_model, acoustic_model, args)
        else:

            model = HKT(text_model, visual_model, acoustic_model,hcf_model, args)

    else:
        raise ValueError("Requested model is not available")

    model.to(DEVICE)
    
    if args.dataset=="mosi":
        loss_fct = torch.nn.MSELoss()
    else:
        loss_fct = BCEWithLogitsLoss()
    

    

    # Prepare optimizer
    # used different learning rates for different componenets.
    
    if args.model == "HKT" :

        if args.include_t == "n":
            other_params = model.get_params()
            optimizer_o,scheduler_o=get_optimizer_scheduler(other_params,num_training_steps,learning_rate=args.learning_rate)

            optimizers=[optimizer_o,]
            schedulers=[scheduler_o,]
        
        else:

        
            text_params,other_params = model.get_params()
            optimizer_o,scheduler_o=get_optimizer_scheduler(other_params,num_training_steps,learning_rate=args.learning_rate)
            optimizer_t,scheduler_t=get_optimizer_scheduler(text_params,num_training_steps,learning_rate=args.learning_rate_t)

            optimizers=[optimizer_o,optimizer_t]
            schedulers=[scheduler_o,scheduler_t]
        
    else:
        params = list(model.named_parameters())

        optimizer_l, scheduler_l = get_optimizer_scheduler(
            params, num_training_steps, learning_rate=args.learning_rate
        )
        
        optimizers=[optimizer_l]
        schedulers=[scheduler_l]
        
        
    return model, optimizers, schedulers,loss_fct




def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    

def main():
    
    wandb.init(project="Fusion_Final_Extra_1", group="early_v_cls_ft_bert_ablation")
    wandb.config.update(args)
    
    if(args.seed == -1):
        seed = random.randint(0, 9999)
        print("seed",seed)
    else:
        seed = args.seed
    
    wandb.config.update({"seed": seed}, allow_val_change=True)
    
    set_random_seed(seed)
    
    train_dataloader,dev_dataloader,test_dataloader=set_up_data_loader()
    print("Dataset Loaded: ",args.dataset)
    num_training_steps = len(train_dataloader) * args.epochs
    
    model, optimizers, schedulers, loss_fct = prep_for_training(
        num_training_steps
    )
    print("Model Loaded: ",args.model)
    if args.dataset == "mosi":
        print("Regression")
        regression=True
    else:
        regression=False
    train(
        model,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        optimizers,
        schedulers,
        loss_fct,
        regression=regression
    )
    

if __name__ == "__main__":
    main()