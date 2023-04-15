import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    
    example["text"] = example["text"].lower()
    return example


### Typos
# For typos, we try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# We randomly select each word with some fixed probability, and replace random letters in that word with one of the 
# nearest keys on the keyboard. We vary the random probablity or which letters to use to achieve the desired accuracy.


### Synonyms
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give us a possible synonym word.
# We randomly select each word with some fixed probability to replace by a synonym.



def custom_transform(example):
    
    typo_prob=0.1
    synonym_prob=0.5

    # Typos
    qwerty_mapping = {
        "a": "qwsz",
        "b": "vghn",
        "c": "xdfv",
        "d": "erfcxs",
        "e": "wrsd",
        "f": "rtgvcd",
        "g": "tyhbvf",
        "h": "yujnbg",
        "i": "ujko",
        "j": "uikmnh",
        "k": "iolmj",
        "l": "opk",
        "m": "njk",
        "n": "bhjm",
        "o": "iklp",
        "p": "ol",
        "q": "wa",
        "r": "edft",
        "s": "awedxz",
        "t": "rfgy",
        "u": "yhji",
        "v": "cfgb",
        "w": "qase",
        "x": "zsdc",
        "y": "tghu",
        "z": "asx"
    }
    
    def generate_typos(word):
        if len(word) <= 1:
            return word
        typo_word = ""
        for i, char in enumerate(word):
            if random.random() < typo_prob:
                if char in qwerty_mapping:
                    candidates = qwerty_mapping[char]
                    typo_char = random.choice(candidates)
                else:
                    typo_char = char
                typo_word += typo_char
            else:
                typo_word += char
        return typo_word
    
    example["text"] = " ".join([generate_typos(word) if random.random() < typo_prob else word for word in example["text"].split()])
    
    nltk.data.path.append("/home/cj2133/nltk_data")
    nltk.download("wordnet")
                  
    # Synonym replacement
    def generate_synonym(word):
        synsets = wordnet.synsets(word)
        if len(synsets) == 0:
            return word
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        synonyms.discard(word)
        if len(synonyms) == 0:
            return word
        return random.choice(list(synonyms))
    
    example["text"] = " ".join([generate_synonym(word) if random.random() < synonym_prob else word for word in example["text"].split()])
    
    return example