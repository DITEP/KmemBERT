class EHRPreprocesser():
    def __init__(self):
        pass

    def __call__(self, text):
        return text.replace("#$", "")