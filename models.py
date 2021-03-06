#https://github.com/vilaysov/IFT6135H19_assignment/blob/master/models.py
from math import ceil 
import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

from exceptions import *

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    '''
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()

    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers. 
    # Your implementation should support any number of stacked hidden layers 
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules 
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
    # provided clones function (as opposed to a regular python list), in order 
    # for Pytorch to recognize these parameters as belonging to this nn.Module 
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.num_layers = num_layers

    self.activation = nn.Tanh()

  def init_weights(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and output biases to 0 (in place). The embeddings should not use a bias vector.
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
    # in the range [-k, k] where k is the square root of 1/hidden_size

  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """

    # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
    return torch.zeros([self.num_layers, self.batch_size,self.hidden_size])

  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    # 
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution, 
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation 
    # function here in order to compute the parameters of the categorical 
    # distributions to be sampled from at each time-step.

    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
   
    return samples

class PyTorch_RNN(nn.Module):
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob,config=None):
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_prob = dp_keep_prob
    self.config = config

    self.decoder = nn.Linear(self.hidden_size, self.vocab_size)

    self.rnn = nn.RNN(input_size=self.emb_size,hidden_size=hidden_size,num_layers=self.num_layers,dropout=1-self.dp_keep_prob)

    if config:
      self.init_weights(config['mean'], config['std'])

  def forward(self,inputs,hidden):
    x_embs = self.embedding(inputs)
    output, h_n = self.rnn(x_embs,hidden)
    return self.decoder(output), h_n

  def init_hidden(self):
    # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
    return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])

  def init_weights(self,mean,std):
    for name, param in rnn.named_parameters():
      if 'weight' in name:
         nn.init.normal(param)


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for 
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()

    # TODO ========================

  def init_weights_uniform(self):
    # TODO ========================

  def init_hidden(self):
    # TODO ========================
    return # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # TODO ========================
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    return samples



class Hierarchical_RNN(nn.Module): # Implements a basic 2 layer hierarchical LSTM.
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn_1 = nn.LSTM(input_size, hidden_size[0], num_layers)    
        self.rnn_2 = nn.LSTM(hidden_size[0], hidden_size[1], num_layers)    

        self.i2o = nn.Linear(hidden_size[1], output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        layer_op = torch.empty([len(input),1,self.hidden_size[0]], requires_grad=False)
        for idx,input_it in enumerate(input):
            hidden_1 = self.initHidden(self.hidden_size[0])
            output_1, hidden_1 = self.rnn_1(input_it,hidden_1)
            layer_op[idx][0] = hidden_1[0].view(self.hidden_size[0])

        hidden_2 = self.initHidden(self.hidden_size[1])
        output_2, hidden_2 = self.rnn_2(layer_op, hidden_2)

        _y = hidden_2[0].view(1,self.hidden_size[1])
        y = self.i2o(_y)
        y = self.softmax(y)
        return output_2,hidden_2,y

    def initHidden(self, size):
        return (torch.zeros(1, 1, size),
                torch.zeros(1, 1, size))


class ClockWork_RNNCell(nn.Module):
    '''
    A Clock RNN partitions the hidden layer into blocks. Each block is assigned a time period Tn. 
    
    To make it more generic, taking the case where blocks are not continous or multiple blocks of 
    size 1 have the same time period:

    For a single trainning example, the hidden layer is a 1-D Tensor of size (hidden_size).
    Each value in this hidden layer has a time period associated with it. Which renders the hidden layer 
    Single Example:
        A list of (hidden_state_value, tn) tuples of size hidden_size. 
        Or In PyTorch terms: (Inserty Answer)
    
    Hidden State Size: H
    Block Size: B
    
    The forward pass works as follows: 
        1. The hidden state gets initilized as a Tensor of size (batch_size,hidden_size)
        2. Each hidden state will be partitoned into blocks rendering a Tensor of size
            if H % B == 0:
                (batch_size, H/B, B)
            else:
                (batch_size,H/B + 1, ceil(B))
        3. Each Tensor along the second dimension (H/B tensors each of size B) will be assigned a time period (tn)
        creating a tensor of size H/B; Tp.
        4. The input will be of dimension (seq_len, batch_size, input_size). Each time step ti renders an input of size 
        (batch_size,input_size). Processing this input for timestep ti happens in two steps:
            We will be processing the hidden_state_weights in both steps after creating a duplicate tensor
            a) Iterating through Tp(j:0->H/B), 
                if ti % Tp[j]:
                    The weights will keep their value
                else:
                    The weights will be initilized with 0
                This will leave only those rows of the weight matrix with their values for which the corresponding Tp[j] % ti == 0
            b) The weight matrix of size (hidden_size,hidden_size) will have ceratin rows as 0
            Considering one row the corresponding hidden state time period is j, blocks in this row of weights corresponding to blocks in the hidden state which have time period less than j will be initilized as 0  
            (and same will be repeated for all rows rendering certain columns as 0)

    '''
    def __init__(self,emb_size, hidden_size,seq_len, batch_size,vocab_size,block_size, config):
      if not block_size:
          raise ClockWork_BlockSizeNone

      if block_size > hidden_size:
          raise ClockWork_BlockSizeError

      self.block_size = block_size
      self.hidden_size = hidden_size
      self.emb_size = emb_size
      self.block_dim = ceil(float(hidden_size)/block_size)


      self.Wx = self.init_weights_uniform(torch.zeros(hidden_size,emb_size))
      self.Wh = self.init_weights_uniform(torch.zeros(hidden_size,hidden_size))


      self.bx = torch.zeros(hidden_size)
      self.bh = torch.zeros(hidden_size)



      if config['activation'] == 'tanh':
        self.activation = nn.Tanh()
      elif config['activation'] == 'relu':
        self.activation = nn.ReLU()
      else:
        raise ActivationError

      self.time_periods = self.assign_time_period(hidden_size, block_size)
      self.modify_recurrent_connections()

    def forward(self, input, hidden):
      '''
      Input will be of shape (seq_len, batch_size, embedding_size)
      '''

      for time_step,_in in enumerate(input, start=1):
        Wh_temp = self.Wh
        Wx_temp = self.Wx
        for row_idx, row in Wh_temp:
          tp = self.time_periods[math.floor(float(row_idx)/self.block_size)]
          if time_step % tp != 0:
            Wh_temp[row_idx][:] = 0
            Wx_temp[row_idx][:] = 0
        hidden = self.activation(F.linear(hidden,Wh_temp,self.bh) + F.linear(_in,Wx_temp))



    def assign_time_period(self, hidden_size,block_size):
      time_periods = []
      for it in range(1,self.block_dim + 1):
        time_periods.append(pow(2,it-1))

      return time_periods

    def modify_recurrent_connections(self):
      '''
      Recurrent connections between a block exist only if the time period of a block is lesser than the block 
      '''
      for row_idx,row in enumerate(self.Wh):
        for con_idx,connection in enumerate(row):
          current_tp = self.time_periods[math.floor(float(row_idx)/self.block_size)]
          connection_tp = self.time_periods[math.floor(float(con_idx)/self.block_size)]

          if connection_tp < current_tp:
            self.Wh[row_idx][con_idx] = 0




    def init_hidden(self):
      # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
      return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])

  def init_weights_uniform(self, tensor, mean=0, std=1):
    return nn.init.normal_(tensor,mean=mean,std=std)


class RNN_AutoEncoder(nn.Module):
  '''
  Basic RNN AutoEncoder
  '''
  def __init__(self,embedding_size, layersizes=[20,10,5], model_type='lstm', vocab_size):
    self.embsize = embedding_size
    self.encoding_layer_sizes = layersizes
    self.decoding_layer_sizes = reversed(layersizes)

    if model_type == 'lstm':
      self.model_type = nn.LSTM
    elif model_type = 'gru':
      self.model_type = nn.GRU
    else:
      self.model_type = nn.RNN

    self.encoder = self.create_encoder()
    self.decoder = self.create_decoder()
    self.output = nn.Linear(self.layer_sizes[0], vocab_size)



  def create_encoder()
    dims_arr = [self.embsize]
    dims_arr.extend(self.layer_sizes)

    return [self.model_type(dims_arr[idx], dims_arr[idx+1], 1) for idx in range(len(dims_arr) - 1)]
  def create_decoder()
    dims_arr = reversed(self.layer_sizes)
    return [self.model_type(dims_arr[idx], dims_arr[idx+1], 1) for idx in range(len(dims_arr) - 1)]

  def initHidden(self, size, batch):
    #Add Hidden state for GRU and RNN
      return (torch.zeros(1, batch, size),
              torch.zeros(1, batch, size))
  
  def forward(_input):
    batch_size = _input.shape[1]
    in_arr = [_input]
    
    #Encoding
    for idx,layer in enumerate(self.encoding_layer_sizes):
      hidden = self.initHidden(layer, batch_size)
      layer_input = in_arr[idx]
      layer_model = self.encoder[idx]
      output, current_state = layer_model(layer_input, hidden)
      in_arr.append(output)

    out_arr = [in_arr[-1]]
    #Decoding:
    for idx, layer in enumerate(self.decoding_layer_sizes):
      hidden = self.initHidden(layer, batch_size)
      layer_input = out_arr[idx]
      layer_model = self.decoder[idx]
      output, current_state = layer_model(layer_input, hidden)
      out_arr.append(output)

    #Returning without applying softmax 
    return self.output(out_arr[-1])











# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill
        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and 
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        return # size: (batch_size, seq_len, self.n_units)






#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

