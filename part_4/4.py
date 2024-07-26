import numpy as np
import pandas as pd
import re
from rdkit import Chem

import codecs
import atomInSmiles
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
from SmilesPE.tokenizer import *

from PIL import Image
from rdkit.Chem import Draw    

import matplotlib.pyplot as plt

import json
from collections import Counter

def tokenize_encoder(token_type, smiles_):
    
    if token_type == 'string':
        pattern = r'(\[.*?\]|Br|Cl|B|C|N|O|F|P|S|I|b|c|n|o|s|\(|\)|\=|\#|\-|\+|\%[0-9]{2}|[0-9]|\.)'
        toks = re.findall(pattern, smiles_)
    
    elif token_type == 'atomwise':
    
    elif token_type == 'kmer':
    
    elif token_type == 'SPE':
    
    elif token_type == 'AIS':
    
    else:
        raise ValueError(f"Unknown token_type: {token_type}")

    return toks


def tokenize_decoder(token_type, toks):

    return smiles

def smiles_to_png(smiles, output_file):

def toks_diff(toks1, toks2):

def count_all_tokens(token_type, data):

def toks_plot(token_type, data, output_file, cutoff):
    
def token_vocabulary(token_type, data, save_vocb = False):


if __name__ == "__main__":

    csv_file_path = '../data/drug/train.csv'
    df = pd.read_csv(csv_file_path)
