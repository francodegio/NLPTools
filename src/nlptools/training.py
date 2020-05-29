""" This module contains everything you need to train a SpaCy NER model.
"""
from typing import List, Union
from datetime import datetime
from tqdm.autonotebook import trange
import numpy as np
import spacy
from spacy.util import minibatch, compounding



def create_blank_ner(train_data, language='es'):
    nlp = spacy.blank(language)
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    return nlp


def train_new_model(
        train_data: list, 
        language = 'es', 
        epochs:int = None, 
        target_gradient: int = None, 
        dropout_rate = 0.1, 
        success_threshold = 0.9, 
        loss_tolerance = None, 
        target_device = 'cpu'):

    if target_device=='gpu':
        spacy.prefer_gpu()
    nlp = create_blank_ner(train_data, language)
    optimizer = nlp.begin_training()
    if not epochs:
        epochs = 300
    
    if not target_gradient:
        min_losses = 1000000
    
    
    progress_bar = trange(epochs, unit='epoch')
    start_time = datetime.now()

    for iteration in progress_bar:
        losses = {}
        batches = minibatch(train_data, size=compounding(4, 64, 1.1))
        
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,
                annotations,
                drop=dropout_rate,
                losses=losses
            )
        progress_bar.set_postfix({"Losses": f"{losses.get('ner'):.2f}"})
        current_losses = np.float(losses.get('ner'))
        
        if iteration == 0:
            start_loss = current_losses
        
        if target_gradient:
            if current_losses < target_gradient:
                finish_time = datetime.now()
                result = nlp
                break
        else:
            if current_losses < min_losses:
                min_losses = current_losses
            elif loss_tolerance and current_losses > min_losses * (1 + loss_tolerance):
                finish_time = datetime.now()
                result = nlp
                break
            elif current_losses < start_loss * (1 - success_threshold) or current_losses < 1:
                finish_time = datetime.now()
                result = nlp

                break
    
    print(f'Total time {finish_time-start_time}')    
    
    return result
            
            