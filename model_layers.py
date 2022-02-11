import torch 
from torch import nn
import torch.nn.functional as F 

from math import sqrt, sin, cos

from config import Config

args = Config()

class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_embedding = TokenEmbeddingSubLayer(args)
        self.PEembedding = PositionalEmbeddingSubLayer(args)
        self.embedding = nn.Sequential(self.token_embedding, self.PEembedding)
        self.dropout_layer = nn.Dropout(args.embedding_dropout_ratio)
        self.pad_id = args.pad_id
        self.device = args.device

    def forward(self, token_tensor, attention_mask = None): # token_tensor :(batch, seq_len)
        embed = self.embedding(token_tensor)
        output = self.dropout_layer(embed)

        if attention_mask == None:
            attention_mask = token_tensor.ne(self.pad_id).long() # masking padding
        return output, attention_mask # output : (batch, seq_len, embed_dim) pad_idxs :(batch_size, 1)

class TokenEmbeddingSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embed_dim
        self.pad_id = args.pad_id

        self.Embedding_layer = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.embedding_dim,
            padding_idx = self.pad_id)
        
    def forward(self, token_tensor, lowest = 1e-13):
        output = self.Embedding_layer(token_tensor)
        output = output * sqrt(self.embedding_dim) + lowest
        return output

class PositionalEmbeddingSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.is_sinusoidal == True:
            PEEmbedding = self.makeSinusiodalEmbedding(args)
            require_grad = False
        else :
            PEEmbedding = torch.rand((args.max_len, args.embed_dim))
            require_grad = True
        self.PEEmbedding = nn.Parameter(PEEmbedding, requires_grad = require_grad) # 토치 변수 선언(for autograd 이용)
    
    def forward(self, token_embedding):
        batch_size, seq_len = token_embedding.shape
        embedding_tensor = self.PEEmbedding[:seq_len, :]
        return token_embedding + embedding_tensor
    
    def makeSinusiodalEmbedding(self, args):
        embedding_tensor = torch.zeros(args.max_len, args.embed_dim)

        even_max = (args.embed_dim + 1)//2
        odd_max = args.embed_dim//2

        for pos in range(args.max_len):
            pos_even = [pos]*even_max
            pos_even = torch.tensor([sin(elem/10000**(2*num/args.embed_dim)) for num, elem in enumerate(pos_even)])
            embedding_tensor[pos, 0::2] = pos_even

            pos_odd = [pos]*odd_max
            pos_odd = torch.tensor([cos(elem/10000**(2*num/args.embed_dim)) for num, elem in enumerate(pos_odd)])
            embedding_tensor[pos, 1::2] = pos_odd

        return embedding_tensor


class MultiHeadSelfAttentionSubLayer(nn.Module): # Q, K, V : (batch_size, =<seq_len, model_dim)
    def __init__(self, args) :
        super().__init__()
        self.ffnn_layer = nn.Linear(args.model_dim, args.embed_dim)
        self.model_dim = args.model_dim
    def forward(self, Q, K, V, attention_mask): # attention mask : (batch_size, seq_len) / Q, K, V : (batch_size, seq_len, model_dim)
        lowest = 1e-13
        attention_score = torch.matmul(Q, K.transpose(1, 2))/sqrt(self.model_dim) + lowest # attention_score : (batch_size, seq_len, seq_len)
        attention_score = self.__attention_masking(attention_score, attention_mask)
        attention_matrix = F.softmax(attention_score, dim = 2) # attention_matrix : (batch_size, seq_len, seq_len)
        attention_result = torch.matmul(attention_matrix, V.transpose(1, 2)) # attention_result : (batch_size, seq_len, model_dim)
        return self.ffnn_layer(attention_result)

    def __attention_masking(self, attention_score, attention_mask, lowest = -1e-13):
        attention_score = attention_score.masked_fill(attention_mask.unsqueeze(1).expand(-1, -1, attention_score.shape[-1]) == 0, lowest)  
        return attention_score

class QKVProjectionSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.QueryProjection = nn.Linear(args.embed_dim, args.model_dim)
        self.KeyProjection = nn.Linear(args.embed_dim, args.model_dim)
        self.ValueProjection = nn.Linear(args.embed_dim, args.model_dim)
    
    def forward(self, query, key, value) : # input_seq : (batch, seq_len, input_dim)
        Q = self.QueryProjection(query) # Q, K, V : (batch, seq_len, model_dim)
        K = self.KeyProjection(key)
        V = self.ValueProjection(value)
        return Q, K, V
        
class ResidualConnectionSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_layer = nn.LayerNorm(args.model_dim)
        self.dropout_layer = nn.Dropout(args.model_dropout_ratio)

    def forward(self, transformed, original) : # transformed : (batch_size, seq_len, embed_dim) original : (batch_size, seq_len, embed_dim)
        dropped = self.dropout_layer(transformed)
        connected = original + dropped # connected : (batch_size, seq_len, embed_dim)
        normalized = self.norm_layer(connected)
        return normalized
    
class LinearTransformSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffnn_dim = args.ffnn_dim
        self.model_dim = args.model_dim
        self.embdding_dim = args.embed_dim

        self.ffnn_layer = nn.Sequential(
            nn.Linear(self.model_dim, self.ffnn_dim),
            nn.ReLU(),
            nn.Linear(self.ffnn_dim, self.embdding_dim)
        )
    
    def forward(self, input_seq) : # input_seq :(batch_size, seq_len, model_dim)
        return self.ffnn_layer(input_seq)


class CrossAttentionSubLayer(nn.Module):
    def __init__(self, args) :
        super().__init__()
        self.ffnn_layer = nn.Linear(args.model_dim, args.embed_dim)
        self.model_dim = args.model_dim

    def forward(self, Q, K, V, query_attention_mask, key_attention_mask):
        lowest  = 1e-13
        attention_score = torch.matmul(Q, K.transpose(1, 2))/sqrt(self.model_dim) + lowest # attention_score_matrix : (batch_size, decoder_seq_len, encoder_seq_len)
        attention_score_matrix = self.__attention_masking(attention_score, query_attention_mask, key_attention_mask)
        attention_matrix = F.softmax(attention_score_matrix, dim = 2)
        attention_result = torch.matmul(attention_matrix, V.transpose(1, 2))
        return self.ffnn_layer(attention_result)

    def __attention_masking(self, attention_score, query_attention_mask, key_attention_mask, lowest = -1e-13):
        attention_score = attention_score.masked_fill(key_attention_mask.unsqueeze(1).expand(-1, attention_score.shape[1], -1) == 0, lowest)
        return attention_score

class MaskedMultiheadSelfAttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffnn_layer = nn.Linear(args.model_dim, args.embed_dim)
        self.model_dim = args.model_dim

    def forward(self, Q, K, V):
        lowest = 1e-13
        attention_score_matrix = torch.matmul(Q, K.transpose(1, 2))/sqrt(self.model_dim) + lowest
        attention_score_matrix = self.__attention_masking(attention_score_matrix)
        attention_matrix = F.softmax(attention_score_matrix, dim = 2)
        attention_result = torch.matmul(attention_matrix, V.transpose(1, 2))
        return self.ffnn_layer(attention_result)

    def __attention_masking(self, attention_score, lowest = -1e-13):
        batch_size, seq_len, seq_len = attention_score.shape
        decoder_attention_mask = torch.range(0, seq_len).unsqueeze(0).expand(seq_len, -1).unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, seq_len, seq_len)의 0 ~ seq_len의 행벡터 반복
        decoder_postion_tensor = torch.range(0, seq_len).unsqueeze(1).expand(-1, seq_len).unsqueeze(0).expand(batch_size, -1, -1) # query 기준 자신의 위치 인덱스 행벡터의 반복
        decoder_attention_mask = decoder_attention_mask.to(attention_score.device)
        decoder_postion_tensor = decoder_postion_tensor.to(attention_score.device)
        attention_score = attention_score.masked_fill(decoder_attention_mask > decoder_postion_tensor, lowest)
        return attention_score

class DecoderHeadLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.embed_dim, args.target_vocab_size)
    
    def forward(self, input_seq) : # input_seq : (batch_size, seq_len, embed_dim)
        transform = self.linear(input_seq)
        return F.log_softmax(transform, dim = 2)