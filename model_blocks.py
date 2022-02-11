import torch
from torch import nn
import torch.nn.functional as F

from config import Config
from model_layers import *

class EncoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.projection = QKVProjectionSublayer(args)
        self.multihead_attention = MultiHeadSelfAttentionSubLayer(args)
        self.residual_block_1 = ResidualConnectionSubLayer(args)
        self.linear_transformation = LinearTransformSublayer(args)
        self.residual_block_2 = ResidualConnectionSubLayer(args)


    def forward(self, input_seq, attention_mask): # input_seq : (batch, seq_len , embed_dim) pad_idxs :(batch_size, 1)
        Q, K, V = self.projection(input_seq, input_seq, input_seq) # Q, K, V :(batch, seq_len, model_dim)
        attention_result = self.multihead_attention(Q, K, V, attention_mask) # attention_result : (batch, seq_len, embedding_dim)
        normalized_result_1 = self.residual_block_1(attention_result, input_seq) # normalized_result_1  : (batch, seq_len, model_dim)
        linear_transformed_result = self.linear_transformation(normalized_result_1) # linear_transformed_result :(batch, seq_len, embdding_dim)
        normalized_result_2 = self.residual_block_2(linear_transformed_result, normalized_result_1) # normalized_result_2 : (batch, seq_len, embdding_dim)
        return normalized_result_2, attention_mask

class DecoderBlock(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.decoder_projection = QKVProjectionSublayer(args)
        self.masked_self_attention = MaskedMultiheadSelfAttentionLayer(args)
        self.residual_block_1 = ResidualConnectionSubLayer(args)
        self.query_projection = QKVProjectionSublayer(args)
        self.cross_attention = CrossAttentionSubLayer(args)
        self.residual_block_2 = ResidualConnectionSubLayer(args)
        self.linear_transformation = LinearTransformSublayer(args)
        self.residual_block_3 = ResidualConnectionSubLayer(args)

    def forward(self, decoder_input, decoder_attention_mask, encoder_output, encoder_attention_mask) : # input_seq : (batch, seq_len, embed_dim)
        Q, K, V = self.projection(decoder_input, decoder_input, decoder_input) # Q, K, V : (batch, seq_len, model_dim)
        masked_self_attention_result = self.masked_self_attention(Q, K, V)
        normalized_result_1 = self.residual_block_1(masked_self_attention_result, decoder_input) # normalized_result_1 : (batch, seq_len, model_dim)
        decoder_query, encoder_key, encoder_value = self.query_projection(normalized_result_1, encoder_output, encoder_output) # decoder_query, encoder_key, encoder_value : (batch, seq_len, model_dim)
        cross_attention_result = self.cross_attention(decoder_query, encoder_key, encoder_value, decoder_attention_mask, encoder_attention_mask)
        normalized_result_2 = self.residual_block_2(cross_attention_result, normalized_result_1)
        linear_transformed_result = self.linear_transformation(normalized_result_2)
        normalized_result_3 = self.residual_block_3(linear_transformed_result, normalized_result_2)
        return normalized_result_3