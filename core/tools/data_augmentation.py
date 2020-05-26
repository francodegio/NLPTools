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
    

def random_date_generator(
    start_year:int=1900, 
    end_year:int=2050, 
    start_month:int=1, 
    end_month:int=12, 
    start_day:int=1, 
    end_day:int=31, 
    mapper:dict=None, 
    seed:int=None
) -> datetime.date:
    if 0 in [start_day, start_month, start_year, end_month, end_year, end_day]:
        warnings.warn('An argument was specified with value 0, returning to default values.')
        mapper = {'start_year':1900, 
                  'end_year':2050, 
                  'start_month':1, 
                  'end_month':12, 
                  'start_day':1, 
                  'end_day':31}
    
    if mapper:
        start_year = mapper['start_year']
        end_year = mapper['end_year']
        start_month = mapper['start_month']
        end_month = mapper['end_month']
        start_day = mapper['start_day']
        end_day = mapper['end_day']
    
    if seed:
        random.seed(seed)
        
    start_date = datetime.date(start_year, start_month, start_day)
    end_date = datetime.date(end_year, end_month, end_day)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    
    return random_date


def date_formatter(date: datetime.date, formality: str = 'random', include_year: bool = True) -> str:
    formality_list = ['basic', 'basic2', 'mixed', 'mixed2', 'regular', 'formal', 'veryformal', 'random']
    
    if formality not in formality_list:
        raise KeyError(f'Keyword `{formality}` not found. Argument `formality` must be one of {formality_list}.')
    elif formality == 'random':
        formality = formality_list[random.randint(0,5)]
    
    month_mapper = {
        '01':'Enero',
        '02':'Febrero',
        '03':'Marzo',
        '04':'Abril',
        '05':'Mayo',
        '06':'Junio',
        '07':'Julio',
        '08':'Agosto',
        '09':'Septiembre',
        '10':'Octubre',
        '11':'Noviembre',
        '12':'Diciembre'}

    day = f'{date.day}' if len(str(date.day)) > 1 else f'0{date.day}'
    month = f'{date.month}' if len(str(date.month)) > 1 else f'0{date.month}'
    
    if formality not in ['basic', 'basic2']:
        day_words = num2words(day, lang='es', to='cardinal') if formality != 'veryformal' else num2words(day, lang='es', to='ordinal')
        month_words = month_mapper[month]
    
    if include_year:
        year = f'{date.year}'    
        year_words = num2words(year, lang='es')
    
    format_mapper = {
        'basic': f'{day}-{month}-{year}' if include_year else f'{day}-{month}',
        'basic2': f'{day}/{month}/{year}' if include_year else f'{day}/{month}',
        'mixed': f'{day} de {month_words} de {year}' if include_year else \
                 f'{day} de {month_words}',
        'mixed2':f'{day} de {month_words} de {year_words}' if include_year else \
                 f'{day} de {month_words}',
        'regular': f'{day_words} de {month_words} de {year_words}' if include_year else \
                   f'{day_words} de {month_words}',
        'formal': f'{day_words} del mes de {month_words} de {year_words}' if include_year\
                  else f'{day_words} del mes de {month_words}',
        'veryformal': f'{day_words} día del mes de {month_words} del año {year_words}' \
                    if include_year else f'{day_words} días del mes de {month_words}'
    }
    result = format_mapper[formality]
    
    return result


def random_name_generator(name_type:str, n:int):
    if n > 20000:
        warnings.warn('Number of samples too big, could cause the application to break. Returning 20.000 samples.')
        n = 20000
    possible_types = ['company', 'person', 'any']
    if not name_type in possible_types:
        raise KeyError(f'{name_type} is not a valid option. Please choose one of the following {possible_types}')
    
    if name_type == 'company':
        names = pd.read_csv('../../data/estatutos/external_sources/companies.csv', dtype=str)['name']
    elif name_type == 'person':
        names = pd.read_csv('../../data/estatutos/external_sources/persons.csv', dtype=str)['name']
    elif name_type == 'any':
        persons = pd.read_csv('../../data/estatutos/external_sources/persons.csv', dtype=str)['name'].sample(10000)
        companies = pd.read_csv('../../data/estatutos/external_sources/companies.csv', dtype=str)['name'].sample(10000)
        names = pd.concat([persons, companies])
        del persons, companies

    result = names.sample(n).to_numpy()
    
    return result


def mandato_generator():
    if np.random.randint(0,3):
        years = np.random.randint(1,11)
        keywords = ['años', 'ejercicios'][np.random.randint(0,2)]
        years_words = num2words(years, lang='es')
        random_year = [years, years_words, f'{years_words} ({years})', f'{years} ({years_words})'][np.random.randint(0,4)]
        result = f'{random_year} {keywords}'
    else:
        result = ['término de duración de la sociedad', 'plazo de duración de la sociedad', 'vencimiento de la sociedad'][np.random.randint(0,3)]
    
    return result


def vigencia_generator():
    years = np.random.randint(1,101)
    years_words = num2words(years, lang='es')
    salad = [years, years_words, f'{years_words} ({years})', f'{years} ({years_words})'][np.random.randint(0,4)]
    
    return salad


def tipicidad_generator():
    company_type = ['sociedad de responsabilidad limitada', 
                    'sociedad anónima',
                    'sociedad por acciones simplificada',
                    'sociedad anónima unipersonal',
                    'sociedad por acciones simplificada unipersonal'][np.random.randint(0, 5)]
    style = ['lower', 'upper', 'title'][np.random.randint(0,3)]
    
    if style == 'lower':
        result = company_type
    elif style == 'upper':
        result = company_type.upper()
    elif style == 'title':
        result = company_type.title()
    
    return result


def id_generator(cuit=False):
    millions = np.random.randint(0,100)
    thousands = np.random.randint(0,1000)
    hundreds = np.random.randint(0,1000)
    
    
    if thousands < 10:
        thousands = f'00{thousands}'
    elif thousands < 100:
        thousands = f'0{thousands}'
    else: 
        thousands = f'{thousands}'

    if hundreds < 10:
        hundreds = f'00{hundreds}'
    elif hundreds < 100:
        hundreds = f'0{hundreds}'
    else: 
        hundreds = f'{hundreds}'
        
    result = f'{millions}.{thousands}.{hundreds}'

    if cuit:
        millions = f'0{millions}' if millions < 10 else f'{millions}'
        beginning = np.random.randint(20,36)
        end = np.random.randint(1,10)
        result = f'{beginning}-{millions}{thousands}{hundreds}-{end}'
    
    return result


