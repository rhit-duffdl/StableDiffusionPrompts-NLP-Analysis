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
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SyntaxOptions, SyntaxOptionsTokens



ptb_to_upos = {
    'VB': 'VERB',   # Verb, base form
    '.': 'PUNCT',   # Punctuation
    'PRP$': 'PRON',  # Possessive pronoun
    'VBG': 'VERB',  # Verb, gerund or present participle
    'RB': 'ADV',    # Adverb
    'DT': 'DET',    # Determiner
    'NNP': 'PROPN',  # Proper noun, singular
    'IN': 'ADP',    # Preposition or subordinating conjunction
    'PRP': 'PRON',  # Personal pronoun
    'RBR': 'ADV',   # Adverb, comparative
    'JJ': 'ADJ',    # Adjective
    'MD': 'AUX',    # Modal
    ',': 'PUNCT',   # Punctuation
    'WRB': 'ADV',   # Wh-adverb
    'NN': 'NOUN',   # Noun, singular or mass
    'VBZ': 'VERB',  # Verb, 3rd person singular present
    'POS': 'PART',  # Possessive ending
    'RP': 'PART',   # Particle
    ':': 'PUNCT',   # Punctuation
    'VBP': 'VERB',  # Verb, non-3rd person singular present
    'CC': 'CCONJ',  # Coordinating conjunction
    'JJR': 'ADJ',   # Adjective, comparative
    'NNS': 'NOUN',  # Noun, plural
    'CD': 'NUM',    # Cardinal number
    'VBD': 'VERB',  # Verb, past tense
    'WP': 'PRON',   # Wh-pronoun
    '$': 'SYM',     # Symbol
    'VBN': 'VERB',  # Verb, past participle
    "''": 'PUNCT',  # Closing quotation mark
    'TO': 'PART',   # 'to'
    '``': 'PUNCT',  # Opening quotation mark
    'WDT': 'PRON',  # Wh-determiner
    'EX': 'PRON',   # Existential 'there'
    'JJS': 'ADJ',   # Adjective, superlative
}


spacy_nlp = spacy.load("en_core_web_sm")
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')




df_train = pd.read_json("data/train.json")
X_train = df_train['sentence'].tolist()[:50]
Y_train = df_train['labels'].tolist()[:50]
Y_train_upos = [[ptb_to_upos.get(label)
                 for label in sentence] for sentence in Y_train]

df_test = pd.read_json("data/dev.json")
X_test = df_test['sentence'].tolist()[:50]
Y_test = df_test['labels'].tolist()[:50]
Y_test_upos = [[ptb_to_upos.get(label)
                for label in sentence] for sentence in Y_test]


def spacy_pos_tagging(text):
    doc = spacy_nlp(text)

    pos_tags = [(token.text, token.pos_) for token in doc]

    return pos_tags


def run_spacy_pos_tagging():
    spacy_pos_tagging_results = []
    spacy_start = time.time()
    for sent in X_train:
        spacy_pos_tagging_results.append(spacy_pos_tagging(" ".join(sent)))
    spacy_end = time.time()
    spacy_duration = spacy_end - spacy_start
    return spacy_duration, spacy_pos_tagging_results


def stanza_pos_tagging(text):

    doc = stanza_nlp(text)

    pos_tags = [(word.text, word.upos)
                for sent in doc.sentences for word in sent.words]

    return pos_tags


def run_stanza_pos_tagging():
    stanza_pos_tagging_results = []
    stanza_start = time.time()
    for sent in X_train:
        stanza_pos_tagging_results.append(stanza_pos_tagging(" ".join(sent)))
    stanza_end = time.time()
    stanza_duration = stanza_end - stanza_start
    return stanza_duration, stanza_pos_tagging_results


def ibm_watson_pos_tagging(text):
    authenticator = IAMAuthenticator(api_key)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url(url)

    response = natural_language_understanding.analyze(
        text=text,
        features=Features(syntax=SyntaxOptions(
            tokens=SyntaxOptionsTokens(part_of_speech=True)))
    ).get_result()

    pos_tags = []
    for token in response['syntax']['tokens']:
        pos_tags.append((token['text'], token['part_of_speech']))

    return pos_tags


def run_ibm_watson_pos_tagging():
    ibm_watson_pos_tagging_results = []
    ibm_watson_start = time.time()
    for sent in X_train:
        ibm_watson_pos_tagging_results.append(
            ibm_watson_pos_tagging(" ".join(sent)))
    ibm_watson_end = time.time()
    ibm_watson_duration = ibm_watson_end - ibm_watson_start
    return ibm_watson_duration, ibm_watson_pos_tagging_results


spacy_dur, spacy_pos_tagging_results = run_spacy_pos_tagging()


def get_accuracy(results):
    total_score = 0
    total = 0
    for i, result in enumerate(results):
        sentence = X_train[i]
        target_labels = Y_train_upos[i]

        # Count the number of words that are in the target labels
        correct_pos = 0
        for (word, label) in zip(sentence, target_labels):
            if (word, label) in result:
                correct_pos += 1

        # Count the number of words that are not in the target labels
        for (word, label) in result:
            if (word, label) not in zip(sentence, target_labels):
                correct_pos -= 1

        total_score += correct_pos / len(sentence)
        total += 1

    return total_score / total


spacy_accuracy = get_accuracy(spacy_pos_tagging_results)
print(f"spaCy Accuracy: {total_score / total}\nspaCy runtime: {spacy_dur}s")


stanza_dur, stanza_pos_tagging_results = run_stanza_pos_tagging()
stanza_accuracy = get_accuracy(stanza_pos_tagging_results)

print(f"Stanza Accuracy: {total_score / total}\nStanza runtime: {stanza_dur}s")

ibm_watson_dur, ibm_watson_pos_tagging_results = run_ibm_watson_pos_tagging()

ibm_watson_accuracy = get_accuracy(ibm_watson_pos_tagging_results)

print(
    f"IBM Watson Accuracy: {total_score / total}\nIBM Watson runtime: {ibm_watson_dur}s")
