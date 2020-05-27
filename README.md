# NLPTools
## A Simple and powerful library for NLP and NER

Intended to be a quick way to implement NLP and NER solutions to the clients.

## Installation
TODO

## Usage
TODO

## Contributing
If you want to contribute to this library, you can find me:
- Email: fdegiovannini@moorea.io
- Mattermost: @fdegiovannini

### Formating
At NLPTools we have an unified formatting for defining functions, classes and docstring.
For docstring, we use the NumPy formatting style with typing. Here's an example of what it looks like:
```python
from typing import Optional, List


def self_explanatory_function_name(                 # Look how arguments have 2 indentations
        well_expressed_arg_1: str, 
        well_expressed_arg_2: bool, 
        well_expressed_arg_3: float = 0.8
    ) -> Optional[List[bool]]:                      # Closing parenthesis have only one indentation.
    """ A very short description of what it does.

    Parameters
    ----------
    well_expressed_arg_1 : str
        A concise yet well described argument.
    well_expressed_arg_2 : bool
        A concise yet well described argument.
    well_expressed_arg_3 : float, optional
        A concise yet well described argument, by default 0.8.

    Returns
    -------
    Optional[List[bool]]
        A well documented behaviour of the output. In this case, 
        possible outputs are a List full of Bool's or None.
    
    Examples
    -------
    Example 1: Examples are optional, but should have this format if provided.
    You are allowed to temporally use NOTE: in the docstring.
    """
    your_operations_and_algorithms = True
    # we do not use comments for any reason.
    # if you have a list comprehension, please use the following style:
    list_comprehension = [
        verbose_object 
        for verbose_object in verbose_list
        if verbose_object is True
    ]
    return list_comprehension
```

Yes, we know. Our docstrings can be longer than our functions. But that's how we maintain it, 
and it is easier for everyone to read the documentation, debug the library and be productive quickly.