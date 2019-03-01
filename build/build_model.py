import torch
import torch.nn as nn
from model.encoderRNN import EncoderRNN
from model.attention_decoderRNN import AttentionDecoderRNN

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def build_model(voc):
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

    # Set checkpoint to load from
    # Set to None if starting from scratch
    load_filename = None
    checkpoint_iter = 4000

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