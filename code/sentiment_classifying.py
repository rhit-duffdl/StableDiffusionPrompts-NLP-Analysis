import pdb
import numpy as np
import pandas as pd
import spacy
from spacy import displacy
import random
import stanza
import time
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
import os
import json
from sklearn.model_selection import train_test_split
from spacy.util import minibatch, compounding
from spacy.training import Example


spacy_nlp = spacy.blank("en")

textcat = spacy_nlp.add_pipe("textcat", last=True)

textcat.add_label("anger")
textcat.add_label("joy")
textcat.add_label("fear")

data = pd.read_csv("data/Emotion_classify_Data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data['Comment'], data['Emotion'], test_size=0.2, random_state=42)

train_data = []
for text, label in zip(X_train, y_train):
    cat = {"cats": {"anger": label == "anger",
                    "joy": label == "joy", "fear": label == "fear"}}
    train_data.append((text, cat))

with spacy_nlp.select_pipes(enable="textcat"):
    optimizer = spacy_nlp.begin_training()
    for i in range(10):
        random.shuffle(train_data)
        losses = {}
        for batch in minibatch(train_data, size=compounding(4., 32., 1.001)):
            examples = []
            for text, annotations in batch:
                doc = spacy_nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            spacy_nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)
        print(f"Losses at iteration {i}: {losses}")


results = []
test_texts = X_test.iloc[:10]
for test_text in test_texts:
    doc = spacy_nlp(test_text)
    results.append((test_text, doc.cats))

with open('data/spacy_sentiment_results.json', 'w') as outfile:
    outfile.write(json.dumps(results, indent=4))
