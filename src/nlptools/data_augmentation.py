""" Here you can find a suite of tools to work with tagged documents. You can render them or 
    create synthetic documents to supersample your dataset.

Classes
-------
- TaggedDoc
    Intended to open tagged documents using the Text Tag Tool.
"""
import warnings
import random
import datetime
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
    - TaggedDoc.save_render

    
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
            


    def _displacy_transform(
            self
        ) -> dict:
        """
        Transforms the entity format from the original document to be used by displacy.

        Returns
        -------
        dict
            A dictionary with the following format:
        {
            'text' : 'the full text of the document',
            'ents' : [
                {
                    'start' : 'where the entity starts in the text',
                    'end' : 'where the entity ends in the text',
                    'label': 'labelname'
                }
            ],
            'title' : 'name of the document' 
        }
        """
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
    
    
    def render(self, style='ent', jupyter=True, page=False, **kwds):
        """ Renders the document title, text and entities in html format using spacy.displacy.render.

        Parameters
        ----------
        style : str, optional
            The type of rendering you want to see, by default 'ent'. For further information, refer to
            spacy.displacy.render.
        jupyter : bool, optional
            If set to True, will render in a jupyter notebook. Otherwise will save the output to html, by default True.
        page : bool, optional
            If set to True, will save an html file., by default False
        """
        for _ in self.displacy_format.get('ents'):
            if not _.get('start') or not _.get('end') or not _.get('label'):
                to_render = {'text': 'Esto es un texto completo con entidades, pero vos no sabes de donde sacarlo.',
                             'ents': [{'start': 46, 'end': 49, 'label': 'GIL'}],
                             'title': 'Por favor leé la documentación.'}
                break
            else:
                to_render = self.displacy_format
                break
        return displacy.render(to_render, style=style, jupyter=jupyter, manual=True, page=page, **kwds)        


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
    
    def save_render(self, filepath:str, **kwds):
        html = displacy.render(self.displacy_format, style='ent', jupyter=False, manual=True, page=True, **kwds)        
        with open(f'{filepath}.html', 'w') as file:
            file.write(html)


def random_date_generator(
        start_year:int=1900, 
        start_month:int=1, 
        start_day:int=1, 
        end_year:int=2050,
        end_month:int=12, 
        end_day:int=31, 
        mapper:dict=None, 
        seed:int=None
    ) -> datetime.date:
    """ Creates a random date in the time span provided. Returns a datetieme.date object.

    Parameters
    ----------
    start_year : int, optional
        The initial year of the target time span, by default 1900.
    start_month : int, optional
        The initial month of the target time span, by default 1.
    start_day : int, optional
        The initial day of the target time span, by default 1.
    end_year : int, optional
        The final year of the target time span, by default 2050.
    end_month : int, optional
        The final month of the target time span, by default 12.
    end_day : int, optional
        The final day of the target time span, by default 31.
    mapper : dict, optional
        A dictionary-like object with same previous arguments as keys and integer as values, by default None.
    seed : int, optional
        The random seed if you're interested in replicating the results, by default None.

    Returns
    -------
    datetime.date
        A random datetime.date object contained in the time span provided.


    Examples
    -------
    >>> from nlptools.data_augmentation import random_date_generator
    >>> random_date_generator()
    datetime.date(1901, 6, 1)
    >>> random_date_generator(1991,2,1, 1991, 2, 2)
    datetime.date(1991, 2, 1)
    >>> random_date_generator(mapper={'start_year': 2000, 'start_month': 1, 'start_day': 1,
    ...                               'end_year': 2025, 'end_month':12, 'end_day':31})
    datetime.date(2010, 10, 9)
    """
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
    
    if formality not in ['basic', 'basic2', 'mixed', 'mixed2']:
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


def capital_generator(style:str='any'):
    millions = np.random.randint(0,2)
    thousands = [x for x in range(0,1000, 5)][np.random.randint(0,200)]
    hundreds = ['000', '500'][np.random.randint(0,2)]
    
    if millions:
        if len(str(thousands)) == 1:
            thousands = f'00{thousands}'
        elif len(str(thousands)) == 2:
            thousands = f'0{thousands}'
        else:
            thousands = str(thousands)
        
        millions = np.random.randint(1,11)
        result = f'{millions}.{thousands}.{hundreds}'
    else:
        result = f'{thousands}.{hundreds}'
    
    if style == 'any':
        style = ['written', 'number', 'mixed'][np.random.randint(0,3)]
    
    if style == 'written':
        result = ''.join(result.split('.'))
        result = num2words(result, lang='es')
    elif style == 'mixed':
        number = ''.join(result.split('.'))
        words = num2words(number, lang='es')
        result = [f'{words} ($ {result})', f'${result} ({words})'][np.random.randint(0,2)]
    elif style == 'number':
        pass
    else:
        raise KeyError('Wrong specification of style. Must be `written`, `number`, `mixed` or `any`')
    return result


def aporte_generator(share_type:str = 'any'):
    thousands = [x for x in range(0,1000, 5)][np.random.randint(0,200)]
    hundreds = ['000', '500'][np.random.randint(0,2)]
    
    if np.random.randint(0,2):
        result = f'{thousands}.{hundreds}'
    else:
        result = thousands
    return result


def address_generator(n:int, legal:bool=False):
    
    df_address = pd.read_csv('../../data/estatutos/external_sources/calles.csv', dtype=str)
    
    if legal:
        result = df_address['departamento'] + ', ' + df_address['provincia']
        result = result.sample(n)
    else:
        altura = pd.Series([np.random.randint(0,5000) for i in range(n*5)])
        
        numeros = [str(np.random.randint(0,10)) for i in range(10)]
        letras = 'A B C D E F G H I J'.split(' ')
        letras_numeros = letras + numeros
        piso = pd.Series([np.random.randint(0,20) for i in range(n*3)])
        tipo = pd.Series([['departamento', 'oficina'][np.random.randint(0,2)] for i in range(n*3)])
        enumeracion = pd.Series([letras_numeros[np.random.randint(0,20)] for i in range(n*3)])
        full = 'piso ' + piso.astype(str) + ', ' + tipo + ' ' + enumeracion.astype(str)
        casa = 'casa ' + piso.astype(str) + ', ' + 'manzana' + ' ' + enumeracion.astype(str)
        full = pd.concat([full, full, casa]).sample(n*3)
        print(len(full))
        connector = [[', de la localidad de ', ', partido de ', ', ', ', departamento de ']\
                     [np.random.randint(0,4)] for i in range(n*3)]
        
        completo =  pd.Series(df_address['nombre'].sample(n*3).str.title().values) + \
                ' ' +\
                pd.Series(altura.sample(n*3).astype(str).values) + \
                ', ' + \
                pd.Series(full.values) + \
                pd.Series(connector) + \
                pd.Series(df_address.sample(n*3)['departamento'].values) + \
                ', ' + \
                pd.Series(df_address.sample(n*3)['provincia'].values)
            
        solo_altura = pd.Series(df_address['nombre'].sample(n*3).str.title().values) + \
                      ' ' + \
                      pd.Series(altura.sample(n*3).astype(str).values) + \
                      ', ' +\
                      pd.Series(df_address['departamento'].sample(n*3).values) + \
                      ', ' +\
                      pd.Series(df_address['provincia'].sample(n*3).values)
        
        esta_ciudad = pd.Series(df_address['nombre'].sample(n).str.title().values) + \
                      ' ' + \
                      pd.Series(altura.sample(n).astype(str).values) + \
                      ', ' +\
                      pd.Series(['de esta ciudad' for i in range(n)])
        
        esta_ciudad_2 = pd.Series(df_address['nombre'].sample(n).str.title().values) + \
                        ' ' +\
                        pd.Series(altura.sample(n).astype(str).values) + \
                        ', ' + \
                        pd.Series(full.sample(n).values) + \
                        pd.Series([', de esta ciudad' for i in range(n)])
        
        
        result = pd.concat([completo, solo_altura, esta_ciudad, esta_ciudad_2]).sample(n).to_list()
    
    return result