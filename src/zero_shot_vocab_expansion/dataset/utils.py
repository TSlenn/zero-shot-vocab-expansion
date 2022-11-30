from nltk.corpus import wordnet


def get_definitions(word):
    """Finds word definitions using wordnet.

    Args:
        word (str): word to search

    Returns:
        list of str: list of definitions. Returns empty list if word is
            not found.
    """
    syns = wordnet.synsets(word)
    return [syn.definition() for syn in syns]
