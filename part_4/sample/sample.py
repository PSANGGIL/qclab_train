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
        toks = atomwise_tokenizer(smiles_)
    elif token_type == 'kmer':
        toks = kmer_tokenizer(smiles_, ngram=4) 
    elif token_type == 'SPE':
        spe_vob= codecs.open('./SPE_ChEMBL.txt')
        spe = SPE_Tokenizer(spe_vob)
        toks = spe.tokenize(smiles_).split(' ') 
    elif token_type == 'AIS':
        toks = atomInSmiles.encode(smiles_).split(' ')
    else:
        raise ValueError(f"Unknown token_type: {token_type}")

    return toks


def tokenize_decoder(token_type, toks):
    
    if token_type in ['string', 'atomwise', 'SPE']:
        smiles = "".join(toks)
    elif token_type == 'kmer':
        smiles = "".join([str(toks[i]) for i in range(len(toks)) if i % 4 == 0])
    elif token_type == 'AIS':
        smiles = coded_smiles = atomInSmiles.decode(" ".join(toks))
    else:
        raise ValueError(f"Unknown token_type: {token_type}")

    return smiles

def smiles_to_png(smiles, output_file):
    molecule = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(molecule)
    img.save(output_file + ".png")

def smi2tok(token_type, data):
    return data.apply(lambda x :tokenize_encoder(token_type, x))

def count_all_tokens(token_type, data):
    data_ = smi2tok(token_type, data)
    all_tokens = [token for sublist in data_ for token in sublist]
    return Counter(all_tokens)

def toks_plot(token_type, data, cutoff):
    token_counts = count_all_tokens(token_type, data)
    sorted_token_counts = dict(sorted(token_counts.items(), key=lambda item: item[1], reverse=True))

    filtered_token_counts = {token: count for token, count in sorted_token_counts.items() if count > cutoff}

    total_count = sum(filtered_token_counts.values())
    normalized_token_counts = {token: count / total_count for token, count in filtered_token_counts.items()}

    plt.figure(figsize=(10, 6))
    plt.bar(normalized_token_counts.keys(), normalized_token_counts.values(), color='skyblue')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.title('Frequency of Tokens in SMILES ' + token_type)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.xticks([])
    
    plt.savefig(token_type + '_frequency.png')

    
def toks_vocabulary(token_type, data, save_vocab = False):
    unique_tokens = sorted(set(count_all_tokens(token_type, data)))  # 중복 제거 및 정렬
    token_dict = {token: idx for idx, token in enumerate(unique_tokens)}
    print(token_dict)
    if save_vocab == True:
        with open(token_type + '_vocabulary.json', 'w' ) as vocab_file:
           json.dump(token_dict , vocab_file)

if __name__ == "__main__":

    csv_file_path = '../data/drug/train.csv'
    df = pd.read_csv(csv_file_path)
    smiles_  = df['SMILES']
    token_type = 'AIS'
    count_all_tokens(token_type, smiles_)

    for token_type in ['string', 'atomwise', 'kmer', 'SPE', 'AIS']:
        toks_vocabulary(token_type, smiles_, True)
        toks_plot(token_type, smiles_, 2)
