""" Here you can find a suite of tools to work with tagged documents. You can render them or 
    create synthetic documents to supersample your dataset.

Classes
-------
- TaggedDoc
    Intended to open tagged documents using the Text Tag Tool.
"""
import warnings
import pandas as pd
import numpy as np
from spacy import displacy
from num2words import num2words


class TaggedDoc:
    """
        Takes a tagged document in dictionary format and creates an object.
    
    Attributes
    --------
    - TaggedDoc.document
        Returns the same dictionary that was provided.
    - TaggedDoc.displacy_ents
        Returns a list of dictionaries with every entity tagged. Each dictionary contains keys: 
        `start`, `end` and `label`.
    - TaggedDoc.title
        Returns the `doc_id` of the document provided.
    - TaggedDoc.text
        Returns the entire text of the document provided.
    - TaggedDoc.text_len
        Returns the lenght of the document measured in characters.
    - TaggedDoc.ents_df
        Returns a pandas.DataFrame object with the entities tagged.

    Methods
    -------
    - TaggedDoc.render
    - TaggedDoc.index_augmentation


    
    """
    def __init__(self, document):
        if not isinstance(document, dict):
            raise TypeError(f'This class only takes a dictionary as input. You provided a {type(document)}.')
        elif [key for key in ['doc_id', 'text', 'entities'] if key not in document.keys()]:
            raise KeyError('The dictionary must contain the keys `doc_id`, `text` and `entities`.')
        else:
            self.document = document
            self.title = document.get('doc_id')
            self.ents_df = pd.DataFrame(document['entities']['tags'], index=[self.title for x in range(len(document['entities']['tags']))]).sort_values('start')
            self.ents = [values.to_dict() for index, values in self.ents_df.iterrows()]
            self.document['entities']['tags'] = self.ents
            self.displacy_format = self._displacy_transform()
            self.displacy_ents = self.displacy_format.get('ents')
            self.text = document.get('text')
            


    def _displacy_transform(self) -> dict:
        ent_list = []
        
        for tag in self.ents:
            ent_list.append(
                    {
                        'start' : tag.get('start'),
                        'end' : tag.get('end'),
                        'label': tag.get('tag')
                    }
                )

        return {
            'text' : self.document.get('text'),
            'ents' : ent_list,
            'title' : self.document.get('doc_id') 
        }
    
    
    def render(self, style='ent', jupyter=True, manual=True, page=False):
        for _ in self.displacy_format.get('ents'):
            if not _.get('start') or not _.get('end') or not _.get('label'):
                to_render = {'text': 'Esto es un texto completo con entidades, pero vos no sabes de donde sacarlo.',
                             'ents': [{'start': 46, 'end': 49, 'label': 'GIL'}],
                             'title': 'Por favor leé la documentación.'}
                break
            else:
                to_render = self.displacy_format
                break
        displacy.render(to_render, style=style, jupyter=jupyter, manual=manual, page=page)        


    def index_augmentation(self):
        new_ents = self.ents_df.copy() # careful with this
        new_ents = new_ents.sort_values('start')
        new_ents.index = range(new_ents.shape[0])
        new_ents['len'] = new_ents.text.apply(len)
        new_ents['new_len'] = new_ents.new_text.apply(len)
        new_ents['diff'] = new_ents['new_len'] - new_ents['len']
        indecis = new_ents.loc[new_ents.diff != 0].index.to_list()

        for index in indecis:
            diff = new_ents.loc[index, 'diff']
            new_ents.loc[index, 'end'] = new_ents.loc[index, 'end'] + diff
            new_ents.loc[index+1:,'start'] = new_ents.loc[index+1:,'start'] + diff
            new_ents.loc[index+1:,'end'] = new_ents.loc[index+1:,'end'] + diff

        return [tagged_entity for tagged_entity in new_ents.T.to_dict().values()]
    

