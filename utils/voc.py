"""
Create a vocabulary and load query/response sentence pairs into memory

Voc class: keeps a mapping from words to indexes,
    a reverse mapping of indexes to words
    a count of each word
    a total word count

    methods:
        addWord: add a word to the vocabulary
        addSentence: add all words in a sentence
        trim: trim infrequently seen words
"""

# Default word tokens
PAD_token = 0   # Used for padding short sentence
SOS_token = 1   # Start-of-sentence token
EOS_token = 2   # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS",
                           EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v>= min_count:
                keep_words.append(k)

        print("keep_words {} / {} = {:.4f}".format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",
                           SOS_token: "SOS",
                           EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens
        for word in keep_words:
            self.addWord(word)
