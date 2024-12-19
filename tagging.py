import spacy


class PosLemmaTagger:
    """
    A class to add part-of-speech tags and lemmas to tokens.
    """

    def __init__(self, language="en_core_web_sm"):
        """
        Initialize the tagger with a spaCy language model.
        """
        try:
            self.nlp = spacy.load(language)
        except OSError:
            raise OSError(
                f"spaCy model '{language}' not found. Please install it by running "
                f"'python -m spacy download {language}'."
            )

    def get_pos_tags(self, text):
        """
        Takes a string input and returns a list of tuples containing tokens and their part-of-speech tags.
        """
        parsed_text = self.nlp(text)
        return [(token.text, token.pos_) for token in parsed_text]

    def get_lemmas(self, text):
        """
        Takes a string input and returns a list of lemmas for each token in the text.
        """
        parsed_text = self.nlp(text)
        return [token.lemma_ for token in parsed_text]
