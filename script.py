import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import os
import codecs
from io import open
import itertools
import math

from utils import load_data, processing


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Un-escape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
lines = load_data.load_lines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = load_data.load_conversations(os.path.join(corpus, "movie_conversations.txt"),
                                   lines, MOVIE_CONVERSATIONS_FIELDS)

# Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter,
                        lineterminator="\n")
    for pair in load_data.extract_sentence_pairs(conversations):
        writer.writerow(pair)

# Print a sample of lines
print("\nSample lines from file:")
load_data.print_lines(datafile)


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = processing.load_prepare_data(corpus_name, datafile, save_dir)
# Print some pair to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

pairs = processing.trim_rare_words(voc, pairs)
