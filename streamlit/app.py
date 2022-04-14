import os
import streamlit as st
import sys

import torch
from transformers import MarianTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import arguments
from src.entities import Sample, create_eval_sample
from src import utils


def prediction(tokenizer, model, source):
    sample = Sample(id=0, source=source)
    sample = create_eval_sample(sample, tokenizer=tokenizer, max_length=args.max_length)
    # prediction
    inputs = torch.tensor([sample['encoding'].tolist()], device=device)
    outputs = model.generate(inputs=inputs, pad_token_id=tokenizer.pad_token_id, output_scores=True,
                                      return_dict_in_generate=True)
    target = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)

    return target, outputs['sequences_scores'][0].item()


def app(tokenizer, model):
    st.title('Machine translation')
    source = st.text_input('English sentence:', help='Please enter only one English sentence',
                           placeholder='Enter the English sentence you want to translate')

    with st.spinner('Wait for it...'):
        target, score = prediction(tokenizer, model, source)

    st.write('**Vietnamese sentence:**', target)
    st.write('**Confidence score:**', score)


if __name__ == '__main__':
    parser = arguments.parse_args()
    args = parser.parse_args()

    # load tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(args.tokenizer_path)
    device = torch.device(utils.get_device())
    model = torch.load(args.saved_model_path, map_location=device)['model']

    app(tokenizer, model)
