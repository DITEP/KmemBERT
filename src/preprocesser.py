'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

class EHRPreprocesser():
    def __init__(self):
        pass

    def __call__(self, text):
        if text:
            return text.replace("#$", "")
        else:
            return ""