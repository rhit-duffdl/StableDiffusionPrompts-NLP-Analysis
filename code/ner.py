import pdb
import numpy as np
import pandas as pd
import spacy
from spacy import displacy
import stanza
import time
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
import os
import json


def my_pretty_print(entities):
    [print(entity) for entity in entities]


# Data from: https://www.kaggle.com/datasets/thedevastator/gustavosta-nlp-research-prompts?select=train.csv
# train_df = pd.read_csv("data/train.csv")
# test_df = pd.read_csv("data/test.csv")
# Seems to me that train and test aren't any different. From the author on Kaggle:
# "The train.csv file contains prompts of varying difficulty levels for natural language processing research"
# "The test.csv file contains difficult prompts for testing natural language processing algorithms"

# Code used to generate sample data. For sake of consistency, I did this once and saved to a CSV
# Feel free to uncomment and run again if you want a different random sample
# sampled = pd.DataFrame(train_df.sample(25).values + test_df.sample(25).values)
# sampled.to_csv("data/sample.csv")


sampled = pd.read_csv("data/sample.csv")
prompt_sentences = sampled['0'].tolist()


spacy_nlp = spacy.load("en_core_web_sm")
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')


def spacy_ner(text):
    doc = spacy_nlp(text)

    entitities = [(entity.text, entity.label_) for entity in doc.ents]

    return entitities


def run_spacy_ner():
    spacy_ner_results = []
    spacy_start = time.time()
    for sent in prompt_sentences:
        spacy_ner_results.append(spacy_ner(sent))
    spacy_end = time.time()
    spacy_duration = spacy_end - spacy_start

    return spacy_duration, spacy_ner_results


def stanza_ner(text):
    doc = stanza_nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.type))

    return entities


def run_stanza_ner():
    stanza_ner_results = []
    stanza_start = time.time()
    for sent in prompt_sentences:
        stanza_ner_results.append(stanza_ner(sent))
    stanza_end = time.time()
    stanza_duration = stanza_end - stanza_start

    return stanza_duration, stanza_ner_results


def ibm_watson_ner(text, api_key):
    authenticator = IAMAuthenticator(api_key)

    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )

    natural_language_understanding.set_service_url(url)

    response = natural_language_understanding.analyze(
        text=text,
        features=Features(entities=EntitiesOptions(limit=50))).get_result()

    entities = [(entity['text'], entity['type'])
                for entity in response['entities']]
    return entities


def run_ibm_watson_ner():
    ibm_watson_ner_results = []
    ibm_watson_start = time.time()
    for sent in prompt_sentences:
        ibm_watson_ner_results.append(ibm_watson_ner(sent, api_key))
    ibm_watson_end = time.time()
    ibm_watson_duration = ibm_watson_end - ibm_watson_start
    return ibm_watson_duration, ibm_watson_ner_results


spacy_dur, spacy_results = run_spacy_ner()
print(f"Spacy: {spacy_dur}s")

stanza_dur, stanza_results = run_stanza_ner()
print(f"Stanza: {stanza_dur}s")

ibm_watson_dur, ibm_watson_results = run_ibm_watson_ner()
print(f"IBM Watson: {ibm_watson_dur}s")


with open('data/spacy_ner_results.json', 'w') as outfile:
    outfile.write(json.dumps(spacy_results, indent=4))

with open('data/stanza_ner_results.json', 'w') as outfile:
    outfile.write(json.dumps(stanza_results, indent=4))

with open('data/ibm_watson_ner_results.json', 'w') as outfile:
    outfile.write(json.dumps(ibm_watson_results, indent=4))
