import torch
from torch import nn
import torch.nn.functional as F

from config import Config
from model_layers import EmbeddingLayer, DecoderHeadLayer
from model_blocks import EncoderBlock, DecoderBlock


class Transformer(nn.Module) :
    def __init__(self, args) -> None:
        super().__init__()
        self.device = args.device
        self.bos_id = args.bos_id
        self.eos_id = args.eos_id
        self.pad_id = args.pad_id

        self.max_len = args.max_len

        self.EncoderEmbedding = EmbeddingLayer(args)
        self.DecoderEmbedding = EmbeddingLayer(args)
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(args) for num_encoder in range(args.num_blocks)])
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(args) for num_decoder in range(args.num_blocks)])
        self.model_head = DecoderHeadLayer(args)
    
    def forward(self, encoder_input_seq, encoder_attention_mask, decoder_input_seq = None, decoder_attention_mask = None) :
        batch_size, encoder_seq_len, _ = encoder_input_seq.shape
        if decoder_input_seq != None:
            decoder_input_seq = decoder_input_seq.shape[1]
        
        embedded_encoder, encoder_attention_mask = self.EncoderEmbedding(encoder_input_seq, encoder_attention_mask)
        encoder_output = self.EncoderBlocks(embedded_encoder, encoder_attention_mask)
        
        if decoder_input_seq: # 훈련 시
            embedded_decoder, decoder_attention_mask = self.DecoderEmbedding(decoder_input_seq, decoder_attention_mask)
            decoder_output = self.DecoderBlocks(embedded_decoder, decoder_attention_mask, encoder_output, encoder_attention_mask)
            return self.model_head(decoder_output)

        else : # inference 시
            decoder_input_seq = torch.full(size = (batch_size, 1), fill_value = self.bos_id, dtype = torch.long, device = self.device)
            decoder_last_step = decoder_input_seq[:, -1]

            while torch.unique(decoder_last_step).tolist() != [self.eos_id, self.pad_id] or decoder_input_seq.shape[-1] < self.max_len :
                embedded_decoder, decoder_attention_mask = self.DecoderEmbedding(decoder_input_seq, decoder_attention_mask)
                decoder_output = self.DecoderBlocks(embedded_decoder, decoder_attention_mask, encoder_output, encoder_attention_mask)
                model_output = self.model_head(decoder_output)
                model_pred = torch.argmax(model_output, dim = 2)
                decoder_last_step = model_pred[:, -1] # 마지막 원소만 가져옴 (batch_size)
                decoder_input_seq = torch.cat((decoder_input_seq, decoder_last_step.unsqueeze(1)), dim = 1)
            return model_output


