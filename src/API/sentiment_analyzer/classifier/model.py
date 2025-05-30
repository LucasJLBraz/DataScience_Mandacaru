import json

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from .sentiment_classifier import SentimentClassifier

with open("config.json") as json_file:
    config = json.load(json_file)



import pickle
from sklearn.preprocessing import LabelEncoder


with open(config["LABEL_ENCODER"], 'rb') as f:
    label_encoder = pickle.load(f)



class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])

        classifier = SentimentClassifier(len(config["CLASS_NAMES"]))
        try:
            classifier.load_state_dict(
                torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
            )
        except:
            print("Error loading model")
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=50,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        # Use the label encoder here
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]


        return (
        predicted_label,
        confidence,
        dict(zip(label_encoder.classes_, probabilities)),
        )


model = Model()


def get_model():
    return model