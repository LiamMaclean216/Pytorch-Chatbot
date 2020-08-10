import pandas as pd
import csv
import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def gen_data(batch_size, gpu = True):
    
    #https://www.kaggle.com/shashankasubrahmanya/preprocessing-cornell-movie-dialogue-corpus
    movie_lines = pd.read_csv('cornell_data/movie_lines.txt', sep = "\+\+\+\$\+\+\+", engine = "python", 
                              index_col = False, names = ["LineID", "Character", "Movie", "Name", "Line"])
    movie_lines = movie_lines[["LineID", "Line"]]

    movie_lines["Line"] = movie_lines['Line'].str.replace('.','')
    movie_lines["Line"] = movie_lines['Line'].str.replace('!','')
    movie_lines["Line"] = movie_lines['Line'].str.replace('?','')
    movie_lines["Line"] = movie_lines['Line'].str.replace('  ',' ')
    movie_lines["Line"] = movie_lines['Line'].str.replace('[^\w\s.!?]','')
    movie_lines["Line"] = movie_lines["Line"].str.lower()

    movie_lines["LineID"] = movie_lines["LineID"].apply(str.strip)
    movie_lines["Line"] = movie_lines["Line"].apply(lambda x : str(x).split(" ")[1:])

    movie_conversations = pd.read_csv("cornell_data/movie_conversations.txt", sep = "\+\+\+\$\+\+\+", 
                                      engine = "python", index_col = False, names =  ["Character1", "Character2", "Movie", "Conversation"])
    movie_conversations = movie_conversations["Conversation"]

    #convert from strings of lists to actual lists
    movie_conversations = movie_conversations.apply(eval)

    word_vecs = pd.read_table("glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    if(not gpu):
        dtype = torch.FloatTensor
    else:
        dtype = torch.cuda.FloatTensor
    
    ba = 0
    vectorized_inputs = []
    vectorized_targets = []
    
    inputs = []
    targets = []
    while True:
        i = random.randint(0, movie_conversations.size - (batch_size + 1))
        batch = movie_conversations.loc[i:i+batch_size].apply(lambda x : movie_lines.loc[(movie_lines['LineID'].isin(x))])
        batch = batch.apply(lambda x : x['Line'].values).values

        for b in batch:
            for idx in range(len(b) - 1):
                vectorized_inputs.append(torch.tensor(word_vecs.loc[b[idx]].values).type(dtype))
                vectorized_targets.append(torch.tensor(word_vecs.loc[b[idx + 1]].values).type(dtype))
               
                inputs.append(b[idx])
                targets.append(b[idx + 1])
            
                ba += 1
                if ba >= batch_size:
                    ba = 0
                    yield (pad_sequence(vectorized_inputs, batch_first = True)
                           , pad_sequence(vectorized_targets, batch_first = True), inputs, targets)
                    vectorized_inputs = []
                    vectorized_targets = []
                    inputs = []
                    targets = []