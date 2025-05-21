import os
import time
import spacy
import json
import onnxruntime as ort
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from torchtext.vocab import build_vocab_from_iterator
from architecture import translate_onnx

def load_vocabularies(models_dir):
    """Load English and German vocabularies from JSON files"""
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

def quantize_onnx_model(input_model_path, output_model_path):
    print(f"Quantizing model {input_model_path} to {output_model_path}")
    try:
        model = onnx.load(input_model_path)
        onnx.checker.check_model(model)
        print(f"Model {input_model_path} is valid")
    except Exception as e:
        print(f"Error validating model: {e}")
        return False
    
    try:
        quantize_dynamic(model_input=input_model_path,model_output=output_model_path,per_channel=False,reduce_range=False,weight_type=QuantType.QUInt8)
        print(f"Model quantized successfully: {output_model_path}")
        return True
    except Exception as e:
        print(f"Error quantizing model: {e}")
        return False

def measure_inference_time(encoder_sess, decoder_sess, sentence, de_nlp, en_vocab, de_vocab, sos_token, eos_token, runs=5):
    translate_onnx(sentence, de_nlp, en_vocab, de_vocab, encoder_sess, decoder_sess, sos_token, eos_token)
    
    total_time = 0
    for _ in range(runs):
        start_time = time.time()
        translate_onnx(sentence, de_nlp, en_vocab, de_vocab, encoder_sess, decoder_sess, sos_token, eos_token)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / runs
    return avg_time

def test_translation_speed(encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, sos_token, eos_token):
    sentence = "Die künstliche Intelligenz hat in den letzten Jahren erhebliche Fortschritte gemacht und wird in vielen Bereichen eingesetzt, von der medizinischen Diagnostik über autonomes Fahren bis hin zur Sprachübersetzung, die wir hier verwenden, um die Leistung unseres quantisierten Modells zu testen."
    long_time = measure_inference_time(encoder_sess, decoder_sess, sentence, de_nlp, en_vocab, de_vocab, sos_token, eos_token)
    
    return long_time

def create_onnx_session(model_path, execution_providers=None):
    if execution_providers is None:
        execution_providers = ['CPUExecutionProvider']
    
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        session = ort.InferenceSession(model_path, providers=execution_providers,sess_options=options)
        return session
    except Exception as e:
        print(f"Error creating session for {model_path}: {e}")
        return None

def main():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    encoder_model_path = os.path.join(models_dir, "encoder.onnx")
    decoder_model_path = os.path.join(models_dir, "decoder.onnx")
    
    encoder_quantized_path = os.path.join(models_dir, "encoder_quant.onnx")
    decoder_quantized_path = os.path.join(models_dir, "decoder_quant.onnx")
    
    if not (os.path.exists(encoder_model_path) and os.path.exists(decoder_model_path)):
        print("Original ONNX models not found. Please ensure they exist.")
        return
    
    try:
        de_nlp = spacy.load('de_core_news_sm')
    except OSError:
        print("Downloading spaCy model for German...")
        os.system("python3 -m spacy download de_core_news_sm")
        de_nlp = spacy.load('de_core_news_sm')
    
    en_vocab, de_vocab = load_vocabularies(models_dir)
    sos_token = '<sos>'
    eos_token = '<eos>'
    
    available_providers = ort.get_available_providers()
    print(f"Available execution providers: {available_providers}")
    
    print("\nLoading original ONNX models...")
    encoder_sess = create_onnx_session(encoder_model_path)
    decoder_sess = create_onnx_session(decoder_model_path)
    
    if encoder_sess is None or decoder_sess is None:
        print("Failed to load original models.")
        return
    
    orig_encoder_size_mb = os.path.getsize(encoder_model_path) / (1024 * 1024)
    orig_decoder_size_mb = os.path.getsize(decoder_model_path) / (1024 * 1024)
    orig_total_size_mb = orig_encoder_size_mb + orig_decoder_size_mb
    
    print("\nTesting original model performance...")
    orig_time = test_translation_speed(encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, sos_token, eos_token)
    
    print("\nQuantizing ONNX models...")
    encoder_success = quantize_onnx_model(encoder_model_path, encoder_quantized_path)
    decoder_success = quantize_onnx_model(decoder_model_path, decoder_quantized_path)
    
    if not (encoder_success and decoder_success):
        print("Failed to quantize models.")
        return
    
    print("\nLoading quantized ONNX models...")
    quant_encoder_sess = create_onnx_session(encoder_quantized_path)
    quant_decoder_sess = create_onnx_session(decoder_quantized_path)
    
    if quant_encoder_sess is None or quant_decoder_sess is None:
        print("Failed to load quantized models.")
        return
    
    quant_encoder_size_mb = os.path.getsize(encoder_quantized_path) / (1024 * 1024)
    quant_decoder_size_mb = os.path.getsize(decoder_quantized_path) / (1024 * 1024)
    quant_total_size_mb = quant_encoder_size_mb + quant_decoder_size_mb
    
    print("\nTesting quantized model performance...")
    quant_time = test_translation_speed(quant_encoder_sess, quant_decoder_sess, de_nlp, en_vocab, de_vocab, sos_token, eos_token)
    
    print("\n=== ONNX MODEL COMPARISON ===")
    print(f"Original Model Size: {orig_total_size_mb:.2f}MB (Encoder: {orig_encoder_size_mb:.2f}MB, Decoder: {orig_decoder_size_mb:.2f}MB)")
    print(f"Quantized Model Size: {quant_total_size_mb:.2f}MB (Encoder: {quant_encoder_size_mb:.2f}MB, Decoder: {quant_decoder_size_mb:.2f}MB)")

    print(f"Original Inference Time: {orig_time:.4f}s")
    print(f"Quantized Inference Time: {quant_time:.4f}s")

if __name__ == "__main__":
    main()
