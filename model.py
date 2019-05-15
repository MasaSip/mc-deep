# -*- coding: utf-8 -*-
"""52_rnn_pohja.ipynb

# Exercise 5. Recurrent neural networks

## Part 2. Mini-batch training of sequence-to-sequence model

In the first part of exercise 5, we updated a sequence-to-sequence model using only one input-ouput sequence pair at a time. That procedure is slow because:
* The update direction is computed using only one example and it is therefore noisy. One needs to use a small learning rate.
* Using only one example in a mini-batch does not fully use the advantage of parallel processing

One difficulty of mini-batch training of sequence-to-sequence is that sequences may have varying lengths and this has to be taken into account. In this exercise, we will implement such training using tools provided by PyTorch.

## Learning goals of part 2

* to learn PyTorch tools for batch processing of sequences with varying lengths
* to learn how to write a custom `DataLoader`
"""

params = [
    {
        'skip_training': False,
        'n_epochs': 1,
        'teacher_forcing_ratio': 0.5,
        'losses_filename': 'exports/t1_losses.csv',
        'losses_plot_filename': 'exports/t1_losses.png',
        'encoder_filename': 'exports/t1_mc_diippi_encoder.pth',
        'decoder_filename': 'exports/t1_mc_diippi_decoder.pth',
        'do_save': 'yes',
        'rap_filename': 'exports/t1_rap.txt',
        'rap_length': 100
    },
     {
        'skip_training': False,
        'n_epochs': 2,
        'teacher_forcing_ratio': 0.5,
        'losses_filename': 'exports/t2_losses.csv',
        'losses_plot_filename': 'exports/t2_losses.png',
        'encoder_filename': 'exports/t2_mc_diippi_encoder.pth',
        'decoder_filename': 'exports/t2_mc_diippi_decoder.pth',
        'do_save': 'yes',
        'rap_filename': 't2_rap.txt',
        'rap_length': 400
    },
     {
        'skip_training': False,
        'n_epochs': 20,
        'teacher_forcing_ratio': 0.5,
        'losses_filename': 'exports/t3_losses.csv',
        'losses_plot_filename': 'exports/t3_losses.png',
        'encoder_filename': 'exports/t3_mc_diippi_encoder.pth',
        'decoder_filename': 'exports/t3_mc_diippi_decoder.pth',
        'do_save': 'yes',
        'rap_filename': 't3_rap.txt',
        'rap_length': 400
    },
     {
        'skip_training': False,
        'n_epochs': 40,
        'teacher_forcing_ratio': 0.5,
        'losses_filename': 'exports/t4_losses.csv',
        'losses_plot_filename': 'exports/t4_losses.png',
        'encoder_filename': 'exports/t4_mc_diippi_encoder.pth',
        'decoder_filename': 'exports/t4_mc_diippi_decoder.pth',
        'do_save': 'yes',
        'rap_filename': 't4_rap.txt',
        'rap_length': 400
    },
     {
        'skip_training': False,
        'n_epochs': 70,
        'teacher_forcing_ratio': 0.5,
        'losses_filename': 'exports/t5_losses.csv',
        'losses_plot_filename': 'exports/t5_losses.png',
        'encoder_filename': 'exports/t5_mc_diippi_encoder.pth',
        'decoder_filename': 'exports/t5_mc_diippi_decoder.pth',
        'do_save': 'yes',
        'rap_filename': 't5_rap.txt',
        'rap_length': 400
    }
]


# Select the device for training (use GPU if you have one)
training_device = 'cpu'


# During evaluation, this cell sets skip_training to True
# skip_training = True

# Select data directory
import os
if os.path.isdir('/coursedata'):
    course_data_dir = '/coursedata'
elif os.path.isdir('./data'):
    course_data_dir = './data'
elif os.path.isdir('../data'):
    course_data_dir = '../data'
else:
    # Specify course_data_dir on your machine
    # course_data_dir = ...
    # YOUR CODE HERE
    from google.colab import drive
    drive.mount('/content/gdrive')
    course_data_dir = '/content/gdrive/My Drive/AaltoDoc/Deep Learning projekti/data/'
    os.chdir('/content/gdrive/My Drive/AaltoDoc/Deep Learning projekti/mc_diippi/')

print('The data directory is %s' % course_data_dir)

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import re
import matplotlib.pyplot as plt
from importlib import reload
    
# class McDiippi:
    # def __init__(self, params):
    #     print("hei")
    #     device = params['device']
    #     print("moioii")
def McDiippi(
    device='cpu', 
    skip_training=False,
    n_epochs=20,
    teacher_forcing_ratio=0.5,
    losses_filename='losses.csv',
    losses_plot_filename='losses.png',
    encoder_filename='mc_diippi_encoder.pth',
    decoder_filename='mc_diippi_decoder.pth',
    do_save='yes',
    rap_filename='rap.txt',
    rap_length=50):

    device = torch.device(device)

    if skip_training:
        # The models are always evaluated on CPU
        device = torch.device(device)

    """## Data

    We use the same translation dataset as in the first part of Exercise 5.
    """

    # Translation data
    try:
        reload(data)
        from data import TranslationDataset, SOS_token, EOS_token, MAX_LENGTH    
    except NameError:
        from data import TranslationDataset, SOS_token, EOS_token, MAX_LENGTH
    
    
    data_dir = os.path.join(course_data_dir, 'translation_data')
    trainset = TranslationDataset(path=data_dir, train=True)

    input_seq, output_seq = trainset[0]
    print('Shapes of input-output sequences:')
    print(input_seq.shape, output_seq.shape)

    print(MAX_LENGTH)

    input_sentence, output_sentence = trainset[np.random.choice(len(trainset))]
    print('Input sentence: "%s"' % ' '.join(trainset.input_lang.index2word[i.item()] for i in input_sentence))
    print('Sentence as tensor of word indices:')
    print(input_sentence)

    print('\nOutput sentence: "%s"' % ' '.join(trainset.output_lang.index2word[i.item()] for i in output_sentence))
    print('Sentence as tensor of word indices:')
    print(output_sentence)

    """## Custom DataLoader

    Next we write a custom data loader which puts sequences of varying lengths in one tensor. We do so by using a custom `collate_fn` as explained [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

    Our collate function combines input sequences in one tensor with extra values filled with zeros. Note that for processing input sequences we are going to use [torch.nn.utils.rnn.PackedSequence](https://pytorch.org/docs/stable/nn.html?highlight=packedsequence#torch.nn.utils.rnn.PackedSequence) class which requires sequences to be sorted by their lengths.

    Similarly, the function combines output sequences in one tensor with extra values filled with zeros. Your task is to implement that.

    Note that:
    * the output sequences need not be sorted by their lengths, so we cannot use [torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence)
    * you should keep the same order of sequences in the input and output tensors
    * the new tensors should have the same data type as input tensors. You can use, for example, function [torch.Tensor.new_full](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.new_full) to create tensors.
    """

    from torch.nn.utils.rnn import pad_sequence

    def collate(list_of_samples):
        """Merges a list of samples to form a mini-batch.

        Args:
        list_of_samples is a list of tuples (input_seq, output_seq),
        input_seq is Tensor([seq_length, 1])
        output_seq is Tensor([seq_length, 1])

        Returns:
        input_seqs: Tensor of padded input sequences: [max_seq_length, batch_size, 1].
        output_seqs: Tensor of padded output sequences: [max_seq_length, batch_size, 1].
        """
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)

        input_seqs, output_seqs = zip(*list_of_samples)
        input_seq_lengths = [len(seq) for seq in input_seqs]
        output_seq_lengths = [len(seq) for seq in output_seqs]
        
        padding_value = 0
        
        # Put all input sequences to one tensor, pad with padding_value
        pad_input_seqs = pad_sequence(input_seqs, batch_first=False, padding_value=padding_value)
        
        batch_size = len(output_seq_lengths)
        max_seq_length = max(output_seq_lengths)
        
        # Put all output sequences to one tensor, pad with padding_value
        # We cannot use pad_sequence because the output sequences are not necessarily sorted according to the lengths
        # YOUR CODE HERE
        typo = type(pad_input_seqs[0,0])
        size = (2,3,1)
        tensor = torch.ones((2,), dtype=pad_input_seqs.dtype)
        pad_output_seqs = tensor.new_full((max_seq_length, batch_size, 1), 0.0)
        for sample, output_seq in enumerate(output_seqs):
            for dimension, value in enumerate(output_seq):
                pad_output_seqs[dimension, sample, padding_value] = value
        return pad_input_seqs, input_seq_lengths, pad_output_seqs, output_seq_lengths

    # Check how collate function combines some test sequences
    # Test with FloatTensors
    pairs = [
        (torch.FloatTensor([1, 2]).view(-1, 1), torch.FloatTensor([3, 4, 5]).view(-1, 1)),
        (torch.FloatTensor([6, 7, 8]).view(-1, 1), torch.FloatTensor([9, 10]).view(-1, 1)),
    ]
    pad_input_seqs, input_seq_lengths, pad_output_seqs, output_seq_lengths = collate(pairs)
    assert pad_input_seqs.shape == torch.Size([3, 2, 1]), "Bad shape of pad_input_seqs: {}".format(pad_input_seqs.shape)
    assert pad_input_seqs.dtype == torch.float32
    assert pad_output_seqs.shape == torch.Size([3, 2, 1]), "Bad shape of pad_output_seqs: {}".format(pad_output_seqs.shape)
    assert pad_output_seqs.dtype == torch.float32
    print('Input sequences combined:')
    print(pad_input_seqs[:, :, 0])
    print('Lengths:', input_seq_lengths)
    print('Output sequences combined:')
    print(pad_output_seqs[:, :, 0])
    print('Lengths:', output_seq_lengths)

    # Test with LongTensors
    pairs = [
        (torch.LongTensor([1, 2]).view(-1, 1), torch.LongTensor([3, 4, 5]).view(-1, 1)),
        (torch.LongTensor([6, 7, 8]).view(-1, 1), torch.LongTensor([9, 10]).view(-1, 1)),
    ]
    pad_input_seqs, input_seq_lengths, pad_output_seqs, output_seq_lengths = collate(pairs)
    assert pad_input_seqs.shape == torch.Size([3, 2, 1]), "Bad shape of pad_input_seqs: {}".format(pad_input_seqs.shape)
    assert pad_input_seqs.dtype == torch.long
    assert pad_output_seqs.shape == torch.Size([3, 2, 1]), "Bad shape of pad_output_seqs: {}".format(pad_output_seqs.shape)
    assert pad_output_seqs.dtype == torch.long
    print("Shapes seem to be ok.")

    # We create custom DataLoader using the implemented collate function
    # We are going to process 64 sequences at the same time (batch_size=64)
    from torch.utils.data import DataLoader
    trainloader = DataLoader(dataset=trainset,
                            batch_size=64,
                            shuffle=True,
                            collate_fn=collate,
                            pin_memory=True)

    """## Encoder

    RNN units implemented in PyTorch such as nn.GRU or nn.LSTM support processing of sequences of varying lenghts. This is done by using function [`torch.nn.utils.rnn.pack_padded_sequence`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence) and [`torch.nn.utils.rnn.pad_packed_sequence`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence). Naturally, the `forward` function of the encoder must process the whole sequence.

    Your task is to implement the `forward` function of the encoder. It should process input sequences in the same way as the encoder from part 1. The difference is that it can process multiple sequences (batch size can be larger than 1).
    You should implement the following steps:
    * embed input sequences
    * pack input sequences using `pack_padded_sequence`
    * apply GRU computations to packed sequences obtained in the previous step
    * convert packed sequence of GRU outputs into padded representation with `pad_packed_sequence`
    """

    class Encoder(nn.Module):
        def __init__(self, dictionary_size, hidden_size):
            super(Encoder, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(dictionary_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)

        def forward(self, pad_seqs, seq_lengths, hidden):
            """
            Args:
            pad_seqs: Tensor [max_seq_length, batch_size, 1]
            seq_lengths: list of sequence lengths
            hidden: Tensor [1, batch_size, hidden_size]

            Returns:
            outputs: Tensor [max_seq_length, batch_size, hidden_size]
            hidden: Tensor [1, batch_size, hidden_size]
            """
            max_seq_length, batch_size, _ = pad_seqs.shape
            X = self.embedding(pad_seqs).squeeze(dim=2)
            X = torch.nn.utils.rnn.pack_padded_sequence(X, seq_lengths)
            X, hidden = self.gru(X)
            X, _ = torch.nn.utils.rnn.pad_packed_sequence(X)
            
            return X, hidden

        def init_hidden(self, batch_size=1, device=device):
            return torch.zeros(1, batch_size, self.hidden_size, device=device)

    # Let's test your code
    hidden_size = 3
    encoder = Encoder(dictionary_size=5, hidden_size=hidden_size).to(device)

    max_seq_length = 4
    batch_size = 2
    hidden = encoder.init_hidden(batch_size=batch_size).to(device)
    pad_seqs = torch.zeros(max_seq_length, batch_size, 1, dtype=torch.int64)
    pad_seqs[:, 0, 0] = torch.tensor([1, 2, 3, 4])
    pad_seqs[:, 1, 0] = torch.tensor([2, 3, 0, 0])
    pad_seqs = pad_seqs.to(device)
    outputs, new_hidden = encoder.forward(pad_seqs=pad_seqs, seq_lengths=[4, 2], hidden=hidden)

    assert outputs.shape == torch.Size([4, batch_size, hidden_size]), \
        "Bad shape of outputs: outputs.shape={}, expected={}".format(
            outputs.shape, torch.Size([4, batch_size, hidden_size]))
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), \
        "Bad shape of outputs: new_hidden.shape={}, expected={}".format(
            new_hidden.shape, torch.Size([1, batch_size, hidden_size]))

    print("The shapes seem to be ok.")

    """## Decoder

    The decoder is similar to the one from part 1 of the exercise, except that it
    * processes multiple sequences at the same time
    * accepts padded target sequences

    Your task is to implement the same functionality as in the decoder of part 1. That is you need to implement the decoder with the following structure:
    <img src="seq2seq_decoder.png" width=500 style="float: left;">

    Similarly to part 1:
    * Apply ReLU nonlinearities to the output word's embeddings (as shown in the figure).
    * We are going to use `nn.NLLLoss` loss for training, which accepts log-probabilities of the target words. Therefore, we need to apply `F.log_softmax` nonlinearity to produce log-probabilities.
    * Use a linear layer to map the states of GRU to the word logits (inputs of `F.log_softmax`).
    """

    class Decoder(nn.Module):
        def __init__(self, hidden_size, output_dictionary_size):
            super(Decoder, self).__init__()
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(output_dictionary_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_dictionary_size)
            self.relu = nn.ReLU()

        def forward(self, hidden, pad_target_seqs=None, teacher_forcing=False):
            """
            Args:
            hidden (tensor):          The state of the GRU (shape [1, batch_size, hidden_size])
            pad_target_seqs (tensor): Tensor of words (word indices) of the target sentence. The shape is
                                        [max_seq_length, batch_size, 1]. If None, the output sequence
                                        is generated by feeding the decoder's outputs (teacher_forcing has to be False).

            Returns:
            outputs (tensor): Tensor of log-probabilities of words in the output language
                                (shape [max_seq_length, batch_size, output_dictionary_size])
            hidden (tensor):  New state of the GRU (shape [1, batch_size, hidden_size])
            """
            if pad_target_seqs is None:
                assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'
            
            _, batch_size, _ = hidden.shape
            prev_word = torch.tensor(SOS_token * np.ones((1, batch_size)), device=device, dtype=torch.int64)
            max_length = pad_target_seqs.size(0) if pad_target_seqs is not None else MAX_LENGTH
            outputs = []  # Collect outputs of the decoder at different steps in this list
            
            for t in range(max_length):
                # YOUR CODE HERE
                previous_vec = self.embedding(prev_word)
                previous_vec = self.relu(previous_vec)
                output, hidden = self.gru(previous_vec, hidden)
                output = self.out(output)
                output = F.log_softmax(output, dim=2)
                outputs.append(output)
                
                if teacher_forcing:
                    # Feed the target as the next input
                    prev_word = pad_target_seqs[t]
                    prev_word = prev_word.unsqueeze(0).squeeze(2)
                else:
                    # Use its own predictions as the next input
                    #print(output[0, :].shape)
                    topv, topi = output[0, :].topk(1)
                    prev_word = topi.squeeze(1).detach()  # detach from history as input
                    prev_word = prev_word.unsqueeze(0)
            outputs = torch.cat(outputs, dim=0)  # [max_length, batch_size, output_dictionary_size]
            
            return outputs, hidden

        def init_hidden(self, batch_size, device=device):
            return torch.rand(1, batch_size, self.hidden_size, device=device)

    # Let's test the shapes
    hidden_size = 2
    output_dictionary_size = 5
    test_decoder = Decoder(hidden_size, output_dictionary_size).to(device)

    max_seq_length = 4
    batch_size = 2
    hidden = test_decoder.init_hidden(batch_size=batch_size, device=device)

    pad_target_seqs = torch.zeros(max_seq_length, batch_size, 1, dtype=torch.int64)
    pad_target_seqs[:, 0, 0] = torch.tensor([1, 2, 3, 4])
    pad_target_seqs[:, 1, 0] = torch.tensor([3, 2, 0, 0])
    pad_target_seqs = pad_target_seqs.to(device)

    outputs, new_hidden = test_decoder.forward(hidden, pad_target_seqs, teacher_forcing=False)
    assert outputs.size(0) <= 4, "Too long output sequence: outputs.size(0)={}".format(outputs.size(0))
    assert outputs.shape[1:] == torch.Size([batch_size, output_dictionary_size]), \
        "Bad shape of outputs: outputs.shape[1:]={}, expected={}".format(
            outputs.shape[1:], torch.Size([batch_size, output_dictionary_size]))
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), \
        "Bad shape of new_hidden: new_hidden.shape={}, expected={}".format(
            new_hidden.shape, torch.Size([1, batch_size, hidden_size]))

    outputs, new_hidden = test_decoder.forward(hidden, pad_target_seqs, teacher_forcing=True)
    assert outputs.shape == torch.Size([4, batch_size, output_dictionary_size]), \
        "Bad shape of outputs: outputs.shape={}, expected={}".format(
            outputs.shape, torch.Size([4, batch_size, output_dictionary_size]))
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), \
        "Bad shape of new_hidden: new_hidden.shape={}, expected={}".format(
            new_hidden.shape, torch.Size([1, batch_size, hidden_size]))

    # Generation mode
    outputs, new_hidden = test_decoder.forward(hidden, None, teacher_forcing=False)
    assert outputs.shape[1:] == torch.Size([batch_size, output_dictionary_size]), \
        "Bad shape of outputs: outputs.shape[1:]={}, expected={}".format(
            outputs.shape[1:], torch.Size([batch_size, output_dictionary_size]))
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), \
        "Bad shape of new_hidden: new_hidden.shape={}, expected={}".format(
            new_hidden.shape, torch.Size([1, batch_size, hidden_size]))

    print('The shapes seem to be ok.')

    hidden_size = 256
    encoder = Encoder(trainset.input_lang.n_words, hidden_size).to(device)
    decoder = Decoder(hidden_size, trainset.output_lang.n_words).to(device)

    """## Loss calculations

    In the training loop (see the code in the training section), the decoder produces a tensor of log-probabilities of words in the output language. We need to use these log-probabilities and the indices of the words in the target sequence to compute the loss. We are going to use [`torch.nn.NLLLoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss) for that.

    Your task is to implement the loss computations in the cell below. You need to compute the average loss for all the words that appear in the **target** sequences:
    $$
    c = \frac{1}{n} \sum_{l=1}^{n} \text{loss}(\mathbf{p}_l, t_l) .
    $$
    Here
    * $n=\sum_k n_k$ is the total number of words in the target sequences ($n_k$ is the number of words in $k$-th target sequence).
    * $\mathbf{p}_l$ is the vector of log-probabilities corresponding to $l$-th word,
    * $t_l$ is the index of the corresponding target word,
    * $\text{loss}$ is implemented by `torch.nn.NLLLoss` (created at the beginning of the cell).

    Note that elements of `pad_target_seqs` equal to `padding_value` indicate elements which should be excluded from the cost computations.
    """

    criterion = nn.NLLLoss(reduction='sum')  # Use this criterion in the loss calculations
    def compute_loss(decoder_outputs, pad_target_seqs, padding_value=0):
        """
        Args:
        decoder_outputs (tensor): Tensor of log-probabilities of words produced by the decoder
                                    (shape [max_seq_length, batch_size, output_dictionary_size])
        pad_target_seqs (tensor): Tensor of words (word indices) of the target sentence (padded with `padding_value`).
                                    The shape is [max_seq_length, batch_size, 1]
        padding_value (int):      Padding value. Keep the default one: the default padding value never
                                    appears in real sequences.
        """    
        
        max_seq_length, batch_size, output_dictionary_size = decoder_outputs.shape
        y = decoder_outputs.view(max_seq_length * batch_size, output_dictionary_size)
        yhat = pad_target_seqs.view(max_seq_length * batch_size, 1).squeeze(1)
        
        n = yhat.shape[0]
        
        mask = (yhat != padding_value)
        #print(y)
        #print("m", mask, "yh", yhat)
        y = y[mask]
        yhat = yhat[mask]
        #print(y)
        #print("m", mask, "yh", yhat)
        output = criterion(y, yhat) / len(y)
        return output

    batch_size = 2
    max_seq_length = 4
    pad_target_seqs = torch.zeros(max_seq_length, batch_size, 1, dtype=torch.int64)
    pad_target_seqs[:, 0, 0] = torch.tensor([1, 2, 3, 4])
    pad_target_seqs[:, 1, 0] = torch.tensor([3, 2, 0, 0])
    decoder_outputs = torch.zeros(max_seq_length, batch_size, 5)
    decoder_outputs[:,:,:] = 1
    loss = compute_loss(decoder_outputs, pad_target_seqs, padding_value=0)

    batch_size = 2
    max_seq_length = 4
    pad_target_seqs = torch.zeros(max_seq_length, batch_size, 1, dtype=torch.int64)
    pad_target_seqs[:, 0, 0] = torch.tensor([1, 2, 3, 4])
    pad_target_seqs[:, 1, 0] = torch.tensor([3, 2, 0, 0])
    decoder_outputs = torch.zeros(max_seq_length, batch_size, 5)
    decoder_outputs[:, 0, [1, 2]] = 1
    decoder_outputs[:, 1, [2]] = 1
    loss = compute_loss(decoder_outputs, pad_target_seqs, padding_value=0)
    assert loss.shape == torch.Size([]), "Bad shape of loss: {}".format(loss.shape)
    print("The shapes seem to be ok.")

    """## Training of sequence-to-sequence model using mini-batches"""


    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.005)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.005)

    """In the training loop, we first encode input sequences using the encoder, then we decode the encoded state using the decoder. As we did previously in part 1, we sometimes use the target output sequence as the input of the decoder (teacher forcing) and sometimes feed the decoder's outputs as inputs (no teacher forcing). Naturally, the latter mode will be used during testing when no target sequence exists.

    The decoder outputs a tensor that contains probabilities of words in the output language. Your task is to use those probabilities to compute the loss. Note that you need to ignore the padded values in the output sequences caused by varying lengths of the output sequences.
    """

    losses = np.array([])

    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = 100  # iterations
        for i, batch in enumerate(trainloader):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths = batch
            batch_size = pad_input_seqs.size(1)
            pad_input_seqs, pad_target_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device)

            encoder_hidden = encoder.init_hidden(batch_size, device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Encode input sequence
            _, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths, encoder_hidden)

            # Decode using target sequence for teacher forcing
            decoder_hidden = encoder_hidden
            teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            decoder_outputs, decoder_hidden = decoder(decoder_hidden, pad_target_seqs, teacher_forcing=teacher_forcing)

            # decoder_outputs is [max_seq_length, batch_size, output_dictionary_size]
            # pad_target_seqs in [max_seq_length, batch_size, 1]
            loss = compute_loss(decoder_outputs, pad_target_seqs, padding_value=0)
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i % print_every) == (print_every-1) or i == (len(trainset) // trainloader.batch_size):
                print('[%d, %5d] loss: %.4f' % (epoch+1, i+1, running_loss/print_every))
                losses = np.append(losses, running_loss / print_every)
                running_loss = 0.0

            if skip_training:
                break
        if skip_training:
            break

    print('Finished Training')

    """Note that training proceeds much faster compared to the implementation in Part 1."""

    if not skip_training:
        np.savetxt(losses_filename, losses)
    else:
        losses = np.genfromtxt(losses_filename)
    plt.figure()
    plt.plot(losses)
    plt.show()
    plt.savefig(losses_plot_filename, dpi=300, format='png')

    # Save the model to disk, submit these files together with your notebook
    
    if not skip_training:
        try:
            if do_save == 'yes':
                torch.save(encoder.state_dict(), encoder_filename)
                torch.save(decoder.state_dict(), decoder_filename)
                print('Model saved to %s, %s.' % (encoder_filename, decoder_filename))
            else:
                print('Model not saved.')
        except:
            raise Exception('The notebook should be run or validated with skip_training=True.')
    else:
        hidden_size = 256
        encoder = Encoder(trainset.input_lang.n_words, hidden_size)
        encoder.load_state_dict(torch.load(encoder_filename, map_location=lambda storage, loc: storage))
        print('Encoder loaded from %s.' % encoder_filename)
        encoder = encoder.to(device)
        encoder.eval()

        decoder = Decoder(hidden_size, trainset.output_lang.n_words)
        decoder.load_state_dict(torch.load(decoder_filename, map_location=lambda storage, loc: storage))
        print('Decoder loaded from %s.' % decoder_filename)
        decoder = decoder.to(device)
        decoder.eval()

    """## Evaluation

    Below is the function that converts an input sequence to an output sequence using the trained sequence-to-sequence model.
    """

    def evaluate(input_seq):
        with torch.no_grad():
            input_length = input_seq.size(0)
            batch_size = 1

            encoder_hidden = encoder.init_hidden(batch_size, device)
            input_seq = input_seq.view(-1, 1, 1).to(device)
            encoder_output, encoder_hidden = encoder(input_seq, [input_length], encoder_hidden)

            decoder_hidden = encoder_hidden
            decoder_outputs, decoder_hidden = decoder(decoder_hidden, pad_target_seqs=None, teacher_forcing=False)

            output_seq = []
            for t in range(decoder_outputs.size(0)):
                topv, topi = decoder_outputs[t].data.topk(1)
                output_seq.append(topi.item())
                if topi.item() == EOS_token:
                    break

        return output_seq

    # Evaluate random sentences from the training set
    print('\nEvaluate on training data:')
    print('-----------------------------')
    for i in range(5):
        input_sentence, target_sentence = trainset[np.random.choice(len(trainset))]
        print('>', ' '.join(trainset.input_lang.index2word[i.item()] for i in input_sentence))
        print('=', ' '.join(trainset.output_lang.index2word[i.item()] for i in target_sentence))
        output_sentence = evaluate(input_sentence)
        print('<', ' '.join(trainset.output_lang.index2word[i] for i in output_sentence))
        print('')

    # Evaluate random sentences from the test set
    testset = TranslationDataset(path=data_dir, train=False)

    if (True):
        
        # Save the input dictionary to a file
        words = list(trainset.input_lang.word2index.keys())
        f = open('input_words.txt','w')
        for w in words:
            f.write(w + '\n')
        f.close()

        # Save the output dictionary to a file
        words = list(trainset.output_lang.word2index.keys())
        f = open('output_words.txt','w')
        for w in words:
            f.write(w + '\n')
        f.close()

    def evaluate(input_seq):
        with torch.no_grad():
            input_length = input_seq.size(0)
            batch_size = 1

            encoder_hidden = encoder.init_hidden(batch_size, device)
            input_seq = input_seq.view(-1, 1, 1).to(device)
            encoder_output, encoder_hidden = encoder(input_seq, [input_length], encoder_hidden)

            decoder_hidden = encoder_hidden
            decoder_outputs, decoder_hidden = decoder(decoder_hidden, pad_target_seqs=None, teacher_forcing=False)

            output_seq = []
            for t in range(decoder_outputs.size(0)):
                topv, topi = decoder_outputs[t].data.topk(1)
                
                #print(type(decoder_outputs[t].data[0]))
                #print(sum(abs(decoder_outputs[t].data[0])))
                results = F.softmax(decoder_outputs[t].data[0], dim=0)
                randi = torch.multinomial(results,1)
                
                output_seq.append(randi.item())
                if topi.item() == EOS_token:
                    break

        return output_seq



    input_line = 'parasta haluun sulle'

    rap = input_line
    input_t = torch.zeros(len(input_line.split()), dtype=torch.int64, device=device)
    for i, w in enumerate(input_line.split()):
        input_t[i] = testset.input_lang.word2index[w]

    # Generate more rap
    for i in range(rap_length):
        next_word = evaluate(input_t)
        if len(next_word[:-1]) != 1:
            print('Expected only one word to be generated')
            print(next_word)
        input_t = torch.cat([input_t,input_t.new_full((1,),next_word[0])])
        rap = rap + ' ' + testset.output_lang.index2word[next_word[0]]

    rap_formated = re.sub('(rivinvaihto *){2,}','\n\n',rap)
    rap_formated = re.sub(' *(rivinvaihto) *', '\n',rap_formated)
    rap_formated = re.sub(' {2,}', ' ',rap_formated)
    print(rap_formated)

    f = open(rap_filename, 'w')
    f.writelines(rap_formated.split('\n'))
    f.close()

    rap_formated = 'moi /n hei'

    print('\nEvaluate on test data:')
    print('-----------------------------')
    for i in range(5):
        input_sentence, target_sentence = testset[np.random.choice(len(testset))]
        print('>', ' '.join(testset.input_lang.index2word[i.item()] for i in input_sentence))
        print('=', ' '.join(testset.output_lang.index2word[i.item()] for i in target_sentence))
        output_sentence = evaluate(input_sentence)
        print('<', ' '.join(testset.output_lang.index2word[i] for i in output_sentence))
        print('')



