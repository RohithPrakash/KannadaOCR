def getCharacter(index):
    """Finds kannada unicode to corresponding label from the recognizer

    Args:

        index (int): Label of the recognized character

    Returns:

        string: Kannada unicode character
    """

    characters = {
        0: u'\u0c85',
        1: u'\u0c86',
        2: u'\u0c87',
        3: u'\u0c88',
        4: u'\u0c89',
        5: u'\u0c8a',
        6: u'\u0c8b',
        7: u'\u0c8e',
        8: u'\u0c8f',
        9: u'\u0c90',
        10: u'\u0c92',
        11: u'\u0c93',
        12: u'\u0c94',
        13: u'\u0c85\u0c82',
        14: u'\u0c85\u0c83',
    }

    return(characters[index])
