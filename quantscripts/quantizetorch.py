import os
import torch
import torch.nn as nn
import json
import time
import spacy
from torchtext.vocab import build_vocab_from_iterator
from architecture import Encoder, Decoder, Seq2Seq, translate_torch

def load_vocabularies(models_dir):
    en_vocab_path = os.path.join(models_dir, "en_vocab.json")
    de_vocab_path = os.path.join(models_dir, "de_vocab.json")
    
    with open(en_vocab_path, 'r', encoding='utf-8') as f:
        en_vocab_dict = json.load(f)
    with open(de_vocab_path, 'r', encoding='utf-8') as f:
        de_vocab_dict = json.load(f)
        
    specials = ['<unk>', '<pad>', '<sos>', '<eos>']
    en_vocab = build_vocab_from_iterator([[]], specials=specials)
    de_vocab = build_vocab_from_iterator([[]], specials=specials)
    
    for token, index in en_vocab_dict.items():
        if token not in specials:
            en_vocab.insert_token(token, index)
    for token, index in de_vocab_dict.items():
        if token not in specials:
            de_vocab.insert_token(token, index)
    
    en_vocab.set_default_index(en_vocab['<unk>'])
    de_vocab.set_default_index(de_vocab['<unk>'])
    
    return en_vocab, de_vocab

def load_model(model_path, device, input_dim, output_dim, embedding_dim=256, hidden_size=512, dropout=0.5):
    state_dict = torch.load(model_path, map_location=device)
    num_layers = 3
    
    encoder = Encoder(input_dim, embedding_dim, hidden_size, num_layers, dropout)
    decoder = Decoder(output_dim, embedding_dim, hidden_size, num_layers, dropout)
    
    model = Seq2Seq(encoder, decoder, device)
    
    try:
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded with strict=False")
    
    model.to(device)
    model.eval()
    
    return model

def quantize_model(model):
    """Apply quantization to the model"""
    print("Quantizing model...")
    model_cp = model.cpu()
    
    if hasattr(torch.backends, 'quantized') and hasattr(torch.backends.quantized, 'supported_engines'):
        engines = torch.backends.quantized.supported_engines
        print(f"Supported quantization engines: {engines}")
        
        if 'fbgemm' in engines:
            print("Using FBGEMM backend")
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in engines:
            print("Using QNNPACK backend")
            torch.backends.quantized.engine = 'qnnpack'
        else:
            print(f"Using {engines[0]} backend")
            torch.backends.quantized.engine = engines[0]
            
        try:
            quantized_model = torch.quantization.quantize_dynamic(model_cp,{nn.LSTM, nn.Linear},dtype=torch.qint8)
            print("Dynamic quantization successful")
            return quantized_model
        except Exception as e:
            print(f"Dynamic quantization error: {e}")
    else:
        print("No quantization engines available")
    
    print("Falling back to half precision (FP16)")
    model_cp.half()
    return model_cp

def measure_inference_time(model, sentence, de_nlp, en_vocab, de_vocab, device, runs=5):
    """Measure inference time for a given sentence"""
    sos_token = '<sos>'
    eos_token = '<eos>'
    
    translate_torch(sentence, model, de_nlp, en_vocab, de_vocab, sos_token, eos_token, device)
    
    total_time = 0
    for _ in range(runs):
        start_time = time.time()
        translate_torch(sentence, model, de_nlp, en_vocab, de_vocab, sos_token, eos_token, device)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / runs
    return avg_time

def test_translation_speed(model, de_nlp, en_vocab, de_vocab, device):

    sentence = "Die künstliche Intelligenz hat in den letzten Jahren erhebliche Fortschritte gemacht und wird in vielen Bereichen eingesetzt, von der medizinischen Diagnostik über autonomes Fahren bis hin zur Sprachübersetzung, die wir hier verwenden, um die Leistung unseres quantisierten Modells zu testen."
    long_time = measure_inference_time(model, sentence, de_nlp, en_vocab, de_vocab, device)
    
    return long_time

def main():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(models_dir, "best.pt")
    quantized_path = os.path.join(models_dir, "best_quantized.pt")
    
    device = torch.device("cpu")
    
    try:
        de_nlp = spacy.load('de_core_news_sm')
    except OSError:
        os.system("python3 -m spacy download de_core_news_sm")
        de_nlp = spacy.load('de_core_news_sm')
    
    en_vocab, de_vocab = load_vocabularies(models_dir)
    
    input_dim = len(de_vocab)
    output_dim = len(en_vocab)
    model = load_model(model_path, device, input_dim, output_dim)
    
    orig_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    orig_long_time = test_translation_speed(model, de_nlp, en_vocab, de_vocab, device)
    
    quantized_model = quantize_model(model)
    
    quant_long_time = test_translation_speed(quantized_model, de_nlp, en_vocab, de_vocab, device)
    
    torch.save(quantized_model, quantized_path)
    
    quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
    
    print("\n=== TORCH MODEL COMPARISON ===")
    print(f"Original Model Size: {orig_size_mb:.2f}MB")
    print(f"Quantized Model Size: {quant_size_mb:.2f}MB")

    print(f"Original Inference Time: {orig_long_time:.4f}s")
    print(f"Quantized Inference Time: {quant_long_time:.4f}s")

if __name__ == "__main__":
    main()
