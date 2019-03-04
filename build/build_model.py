import os
import torch
import torch.nn as nn
from model.encoderRNN import EncoderRNN
from model.attention_decoderRNN import AttentionDecoderRNN
from torch import optim
from procedure import train_procedure
from model.greedy_decoder import GreedySearchDecoder
from procedure import evaluate

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Configure models
model_name = 'chatbot_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64


def build_model(corpus_name, save_dir, pairs, voc):
    # Set checkpoint to load from
    # Set to None if starting from scratch
    checkpoint_iter = 4000
    load_filename = os.path.join(save_dir, model_name, corpus_name,
                                 '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                 '{}_checkpoint.tar'.format(checkpoint_iter))
    if not os.path.isfile(load_filename):
        load_filename = None

    # Load model if a load_filename is provided
    if load_filename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_filename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)

    if load_filename:
        embedding.load_state_dict(embedding_sd)

    # Initialize encoder decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = AttentionDecoderRNN(attn_model, embedding, hidden_size, voc.num_words,
                                  decoder_n_layers, dropout)
    if load_filename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Model built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if load_filename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    train_procedure.train_iters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
                                decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
                                save_dir, n_iteration, batch_size, print_every, save_every, clip,
                                corpus_name, load_filename, hidden_size)


def load_model(corpus_name, save_dir, voc):
    # Set checkpoint to load from
    # Set to None if starting from scratch
    checkpoint_iter = 4000
    load_filename = os.path.join(save_dir, model_name, corpus_name,
                                 '{}-{}-{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                 '{}_checkpoint.tar'.format(checkpoint_iter))
    if not os.path.isfile(load_filename):
        load_filename = None

    # Load model if a load_filename is provided
    if load_filename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_filename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)

    if load_filename:
        embedding.load_state_dict(embedding_sd)

    # Initialize encoder decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = AttentionDecoderRNN(attn_model, embedding, hidden_size, voc.num_words,
                                  decoder_n_layers, dropout)
    if load_filename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Model built and ready to go!')

    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)
    # Begin chatting (uncomment and run the following line to begin)
    evaluate.evaluate_input(searcher, voc)
