from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import torch
from torch.utils.data import Dataset

from config import Config

args = Config()


class CustomDataset(Dataset):
    def __init__(self, kor_dir, eng_dir, korean_tokenizer, english_tokenzier, args) -> None:
        super().__init__() 
        self.target = read_corpus(eng_dir)
        self.source = read_corpus(kor_dir)
        
        self.korean_tokenizer = korean_tokenizer
        self.english_tokenizer = english_tokenzier

        assert len(self.target) == len(self.source), "데이터셋 길이가 다릅니다. 확인해보세요."

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index) :
        source_instance, target_instance = self.source[index], self.target[index]
        source = self.korean_tokenizer.encode(source_instance)
        target = self.english_tokenizer.encode(target_instance)
        return (source, target)

def collate_fn(batch):
    source_tensor = torch.full(size = (len(batch), args.max_len), fill_value = args.pad_id)
    target_tensor = torch.full(size = (len(batch), args.max_len + 1), fill_value = args.pad_id)

    source_mask = torch.full(size = (len(batch), args.max_len), fill_value = 0)
    target_mask = torch.full(size = (len(batch), args.max_len + 1), fill_value = 0)
    max_len = args.max_len

    for num, (source, target) in enumerate(batch):
        source_preprocessing = torch.tensor(source)[:max_len]
        source_len = source_preprocessing.shape[0]
        target_preprocessing = torch.tensor(target)[:max_len + 1]      
        target_len = target_preprocessing.shape[0]
        source_tensor[num, :source_len] = source_preprocessing
        target_tensor[num, :target_len] = target_preprocessing

        source_mask[num, :source_len] = 1
        target_mask[num, :target_len-1] = 1
    return (source_tensor, source_mask), (target_tensor, target_mask)


def read_corpus(file_dir) : 
    with open(file_dir, encoding = "utf-8")as f:
        corpus = f.readlines()
        corpus = [sentence.replace("\n", "") for sentence in corpus]
    return corpus

def build_tokenizer(corpus_dir, save_dir, vocab_size=args.vocab_size, min_freq=args.min_freq):
    corpus = read_corpus(corpus_dir)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[MASK]"], 
        vocab_size = vocab_size, 
        min_frequency = min_freq
        )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(corpus, trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ]
        )
    tokenizer.save(save_dir)
    return tokenizer

def load_tokenizer(tokenizer_dir):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.load(tokenizer_dir)
    return tokenizer