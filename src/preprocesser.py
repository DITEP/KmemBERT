'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

class EHRPreprocesser():
    def __init__(self, max_words=None):
        self.max_words = max_words


    def __call__(self, text):
        text = text.replace("#$", "")
        if self.max_words:
            words = text.split()
            if len(words)>max_words:
                return " ".join(words[:self.max_words//2]+["..."]+words[-self.max_words//2:])

        return text
