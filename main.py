from config import Config
from data import build_tokenizer, load_tokenizer, CustomDataset, collate_fn
from models import Transformer
from utils import train

from torch.utils.data import DataLoader

args = Config()

eng_tokenzier = build_tokenizer(
    corpus_dir = f"{args.data_dir}/train.en", 
    save_dir = args.tokenizer_dir,
    vocab_size = args.vocab_size, 
    min_freq = args.min_freq
    )

kor_tokenizer = build_tokenizer(
    corpus_dir = f"{args.data_dir}/train.ko", 
    save_dir = args.tokenizer_dir,
    vocab_size = args.vocab_size, 
    min_freq = args.min_freq
    )

train_dataset = CustomDataset(
    kor_dir = f"{args.data_dir}/train.ko",
    eng_dir = f"{args.data_dir}/train.en",
    korean_tokenizer = kor_tokenizer,
    english_tokenzier = eng_tokenzier,
    args = args
    )

valid_dataset = CustomDataset(
    kor_dir = f"{args.data_dir}/valid.ko",
    eng_dir = f"{args.data_dir}/valid.en",
    korean_tokenizer = kor_tokenizer,
    english_tokenzier = eng_tokenzier,
    args = args
    )

train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size = args.valid_batch_size, shuffle = True, collate_fn = collate_fn)
model = Transformer(args)


def main():
    train(model, train_dataloader, valid_dataloader, eng_tokenzier, args)

if __name__ == "__main__":
    main()