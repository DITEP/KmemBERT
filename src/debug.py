from .config import Config
from .health_bert import HealthBERT


config = Config({})
config.resume = "training_21-03-11_132_epochs"


model = HealthBERT("cpu", config)
print(len(model.tokenizer))