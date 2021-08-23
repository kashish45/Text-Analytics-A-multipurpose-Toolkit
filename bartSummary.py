from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import os
 
# os.environ['http_proxy'] = "http://ukproxy.bip.uk.fid-intl.com:8000"
# os.environ['https_proxy'] = "http://ukproxy.bip.uk.fid-intl.com:8000"
tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def bart_model(text):
    inputs = tokenizer.batch_encode_plus([text],return_tensors='pt')
    encoded_input_trc={}
    for k,v in inputs.items():
        v_truncated = v[:,:1000]
        encoded_input_trc[k]=v_truncated
    summary_ids = model.generate(encoded_input_trc['input_ids'], early_stopping=True)
    bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return bart_summary