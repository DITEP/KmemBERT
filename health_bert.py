import json

from transformers import CamembertForSequenceClassification, CamembertTokenizer

class HealthBERT:
    def __init__(self):
        pass

    def get_model_tokenizer(voc_path = None, model_name = "camembert-base"):
        camembert = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = CamembertTokenizer.from_pretrained(model_name)

        if not voc_path:
            return camembert, tokenizer

        else:
            with open(voc_path) as json_file:
                voc_list = json.load(json_file)
            new_tokens = tokenizer.add_tokens([ token for (token, _) in voc_list ])
            print(f"Added {new_tokens} tokens to the tokenizer")

            new_dim = new_tokens + camembert.get_input_embeddings().num_embeddings
            camembert.resize_token_embeddings(new_dim)

            return camembert, tokenizer

