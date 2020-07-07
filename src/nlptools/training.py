""" This module contains everything you need to train a SpaCy NER model.
"""
from typing import List, Union
from datetime import datetime
from nlptools.comparison import is_similar_word
from tqdm.autonotebook import trange
import numpy as np
import spacy
from spacy.util import minibatch, compounding



def create_blank_ner(
        train_data, 
        language='es'
    ) -> spacy.lang:
    """ Creates a new nlp model with only one object in pipeline, called ner.

    Parameters
    ----------
    train_data : [type]
        The data required to train the model.
    language : str, optional
        The language of the model you want to train, by default 'es'.

    Returns
    -------
    spacy.lang.es.Spanish
        An spacy object to process the text to extract entities.
    """
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
        target_device = 'cpu'
    ) -> spacy.lang:
    """ Build a new blank spacy model and trains it with the entities provided.

    Parameters
    ----------
    train_data : list
        The data required to train the model.
    language : str, optional
        The language of the model you want to train, by default 'es'.
    epochs : int, optional
        The number of times you want to show the data to the model. If set to None, 
        will iterate 300 times or will cut when it finds the best possible model, 
        given the hyper-parameters. By default None.
    target_gradient : int, optional
        The expected level of the gradient to finish it's training, by default None.
    dropout_rate : float, optional
        How much of the data learned you want to force to throw each iteration 
        to avoid overffiting, by default 0.1
    success_threshold : float, optional
        A percentage of expected minimization of the gradient, by default 0.9
    loss_tolerance : [type], optional
        A threshold to avoid catastrophic forgetting, by default None
    target_device : str, optional
        Whether to train on cpu or gpu, if available, by default 'cpu'.

    Returns
    -------
    spacy.lang.es.Spanish
        A trained model capable to recognize the target entities to a certain extent.
    """

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
    finish_time = None
    result = None

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
        result = nlp
        
        if iteration == 0:
            start_loss = current_losses
        
        if target_gradient:
            if current_losses < target_gradient:
                finish_time = datetime.now()
                break
        else:
            if current_losses < min_losses:
                min_losses = current_losses
            elif loss_tolerance and current_losses > min_losses * (1 + loss_tolerance):
                finish_time = datetime.now()
                break
            elif current_losses < start_loss * (1 - success_threshold) or current_losses < 1:
                finish_time = datetime.now()
                break

    if not finish_time:
        finish_time = datetime.now()                
    
    print(f'Total time {finish_time-start_time}')    
    
    return result