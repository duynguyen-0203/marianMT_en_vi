import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments of machine translation app', allow_abbrev=False)
    parser.add_argument('--saved_model_path', type=str, default='training/2022-04-13_14-23-40.330202/bestModel.pt')
    parser.add_argument('--tokenizer_path', type=str, default='Helsinki-NLP/opus-mt-en-vi')
    parser.add_argument('--max_length', type=int, default=64)

    return parser
