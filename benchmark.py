import os
import time
import json
import torch
import torch.profiler
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import gc
import spacy
from torchtext.vocab import build_vocab_from_iterator
from architecture import Encoder, Decoder, Seq2Seq

NUM_RUNS = 20
WARMUP_RUNS = 2
MAX_LEN = 25
BATCH_SIZE = 32
DEVICE = torch.device("cpu")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_vocab():
    """Load vocabularies from files using torchtext's Vocab class"""
    en_vocab_path = os.path.join(MODELS_DIR, "en_vocab.json")
    de_vocab_path = os.path.join(MODELS_DIR, "de_vocab.json")
    
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

def load_torch_model(model_path, input_dim, output_dim):
    print(f"Loading PyTorch model from {model_path}")
    
    is_quantized = "quant" in model_path.lower()
    
    if is_quantized:
        try:
            torch.backends.quantized.engine = 'qnnpack'        
            model = torch.load(model_path, map_location=DEVICE)
            return model
        except Exception as e:
            print(f"Failed to load quantized model : {e}")
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        num_layers = 3
        EMBEDDING_DIM = 256
        HIDDEN_SIZE = 512
        DROPOUT = 0.2
        encoder = Encoder(input_dim, EMBEDDING_DIM, HIDDEN_SIZE, num_layers, DROPOUT)
        decoder = Decoder(output_dim, EMBEDDING_DIM, HIDDEN_SIZE, num_layers, DROPOUT)
        model = Seq2Seq(encoder, decoder, DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def load_onnx_sessions(encoder_path, decoder_path):
    providers = ['CPUExecutionProvider']
    
    encoder_session = ort.InferenceSession(encoder_path, providers=providers)
    decoder_session = ort.InferenceSession(decoder_path, providers=providers)
    
    return encoder_session, decoder_session

def translate_torch(sentence, model, de_nlp, en_vocab, de_vocab, sos_token, eos_token):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]

        tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = [de_vocab[token] for token in tokens]
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(DEVICE)
        hidden, cell = model.encoder(tensor)
        inputs = [en_vocab[sos_token]]
        for _ in range(MAX_LEN):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(DEVICE)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens

def translate_onnx(sentence, encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, sos_token, eos_token):
    tokens = [sos_token] + [tok.text.lower() for tok in de_nlp.tokenizer(sentence)] + [eos_token]
    input_ids = [de_vocab[token] for token in tokens]
    src_tensor = np.array(input_ids, dtype=np.int64).reshape(-1, 1) 

    encoder_outputs = encoder_sess.run(None, {"src": src_tensor})
    hidden, cell = encoder_outputs

    inputs = [en_vocab[sos_token]]
    translated_tokens = [sos_token]

    for _ in range(MAX_LEN):
        input_token = np.array([inputs[-1]], dtype=np.int64)
        
        decoder_outputs = decoder_sess.run(
            None,
            {
                "input_token": input_token,
                "hidden": hidden,
                "cell": cell
            }
        )
        
        output, hidden, cell = decoder_outputs
        pred_token_id = int(np.argmax(output, axis=1)[0])
        inputs.append(pred_token_id)
        token = en_vocab.lookup_token(pred_token_id)
        translated_tokens.append(token)
        
        if pred_token_id == en_vocab[eos_token]:
            break

    return translated_tokens

def batch_translate_torch(sentences, model, de_nlp, en_vocab, de_vocab, sos_token, eos_token):
    model.eval()
    translated_results = []
    
    with torch.no_grad():
        for sentence in sentences:
            if isinstance(sentence, str):
                tokens = [token.text for token in de_nlp.tokenizer(sentence)]
            else:
                tokens = [token for token in sentence]

            tokens = [token.lower() for token in tokens]
            tokens = [sos_token] + tokens + [eos_token]
            ids = [de_vocab[token] for token in tokens]
            tensor = torch.LongTensor(ids).unsqueeze(-1).to(DEVICE)
            hidden, cell = model.encoder(tensor)
            inputs = [en_vocab[sos_token]]
            for _ in range(MAX_LEN):
                inputs_tensor = torch.LongTensor([inputs[-1]]).to(DEVICE)
                output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
                predicted_token = output.argmax(-1).item()
                inputs.append(predicted_token)
                if predicted_token == en_vocab[eos_token]:
                    break
            tokens = en_vocab.lookup_tokens(inputs)
            translated_results.append(tokens)
    
    return translated_results

def batch_translate_onnx(sentences, encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, sos_token, eos_token):
    translated_results = []
    
    for sentence in sentences:
        tokens = [sos_token] + [tok.text.lower() for tok in de_nlp.tokenizer(sentence)] + [eos_token]
        input_ids = [de_vocab[token] for token in tokens]
        src_tensor = np.array(input_ids, dtype=np.int64).reshape(-1, 1) 

        encoder_outputs = encoder_sess.run(None, {"src": src_tensor})
        hidden, cell = encoder_outputs

        inputs = [en_vocab[sos_token]]
        translated_tokens = [sos_token]

        for _ in range(MAX_LEN):
            input_token = np.array([inputs[-1]], dtype=np.int64)
            
            decoder_outputs = decoder_sess.run(
                None,
                {
                    "input_token": input_token,
                    "hidden": hidden,
                    "cell": cell
                }
            )
            
            output, hidden, cell = decoder_outputs
            pred_token_id = int(np.argmax(output, axis=1)[0])
            inputs.append(pred_token_id)
            token = en_vocab.lookup_token(pred_token_id)
            translated_tokens.append(token)
            
            if pred_token_id == en_vocab[eos_token]:
                break

        translated_results.append(translated_tokens)
    
    return translated_results

def benchmark_torch_model(model, test_sentences, en_vocab, de_vocab, de_nlp, name):
    print(f"Benchmarking {name} - Single Inference...")
    
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    
    latencies = []
    
    for _ in range(WARMUP_RUNS):
        translate_torch(test_sentences[0], model, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
    
    gc.collect()
    
    profiler_activities = [torch.profiler.ProfilerActivity.CPU]

    with torch.profiler.profile(
        activities=profiler_activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        for sentence in test_sentences:
            start_time = time.time()
            translate_torch(sentence, model, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000) 
    
    prof_stats = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    
    return {
        "name": f"{name} (Single)",
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies),
        "prof_stats": prof_stats
    }

def benchmark_batch_torch_model(model, test_sentences, en_vocab, de_vocab, de_nlp, name):
    print(f"Benchmarking {name} - Batch Inference (batch size={BATCH_SIZE})...")
    
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    
    latencies = []
    
    batches = []
    for i in range(0, len(test_sentences), BATCH_SIZE):
        batch = test_sentences[i:i+BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            batch = batch + [batch[0]] * (BATCH_SIZE - len(batch))
        batches.append(batch)
    
    while len(batches) < NUM_RUNS:
        batches.append(batches[0])
    
    for _ in range(WARMUP_RUNS):
        batch_translate_torch(batches[0], model, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
    
    gc.collect()

    for batch in batches[:NUM_RUNS]:
        start_time = time.time()
        batch_translate_torch(batch, model, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
        end_time = time.time()
        
        latencies.append((end_time - start_time) * 1000) 
    
    return {
        "name": f"{name} (Batch)",
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies)
    }

def benchmark_onnx_model(encoder_sess, decoder_sess, test_sentences, en_vocab, de_vocab, de_nlp, name):
    print(f"Benchmarking {name} - Single Inference...")
    
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    
    latencies = []
    
    for _ in range(WARMUP_RUNS):
        translate_onnx(test_sentences[0], encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
    
    gc.collect()
    
    profiler_activities = [torch.profiler.ProfilerActivity.CPU]
    
    with torch.profiler.profile(
        activities=profiler_activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        for sentence in test_sentences:
            start_time = time.time()
            translate_onnx(sentence, encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000) 
    
    prof_stats = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    
    return {
        "name": f"{name} (Single)",
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies),
        "prof_stats": prof_stats
    }

def benchmark_batch_onnx_model(encoder_sess, decoder_sess, test_sentences, en_vocab, de_vocab, de_nlp, name):
    print(f"Benchmarking {name} - Batch Inference (batch size={BATCH_SIZE})...")
    
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    
    latencies = []
    
    batches = []
    for i in range(0, len(test_sentences), BATCH_SIZE):
        batch = test_sentences[i:i+BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            batch = batch + [batch[0]] * (BATCH_SIZE - len(batch))
        batches.append(batch)
    
    while len(batches) < NUM_RUNS:
        batches.append(batches[0])
    
    for _ in range(WARMUP_RUNS):
        batch_translate_onnx(batches[0], encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
    
    gc.collect()
    
    for batch in batches[:NUM_RUNS]:  
        start_time = time.time()
        batch_translate_onnx(batch, encoder_sess, decoder_sess, de_nlp, en_vocab, de_vocab, SOS_TOKEN, EOS_TOKEN)
        end_time = time.time()
        
        latencies.append((end_time - start_time) * 1000)  
    
    return {
        "name": f"{name} (Batch)",
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies)
    }

def plot_results(results):
    single_results = [result for result in results if "(Single)" in result["name"]]
    batch_results = [result for result in results if "(Batch)" in result["name"]]
    
    single_names = [result["name"].replace(" (Single)", "") for result in single_results]
    single_latencies = [result["latency"] for result in single_results]
    single_latency_stds = [result.get("latency_std", 0) for result in single_results]
    
    batch_names = [result["name"].replace(" (Batch)", "") for result in batch_results]
    batch_latencies = [result["latency"] for result in batch_results]
    batch_latency_stds = [result.get("latency_std", 0) for result in batch_results]
    
    plt.figure(figsize=(15, 8))
    
    width = 0.35
    x = np.arange(len(single_names))
    
    plt.bar(x - width/2, single_latencies, width, label='Single Inference', 
            yerr=single_latency_stds, alpha=0.8, color='blue')
    plt.bar(x + width/2, batch_latencies, width, label=f'Batch Inference (size={BATCH_SIZE})', 
            yerr=batch_latency_stds, alpha=0.8, color='orange')
    
    plt.ylabel('Latency (ms)')
    plt.title('Inference Latency Comparison')
    plt.xticks(x, single_names, ha='right')
    plt.legend()
    
    for i, v in enumerate(single_latencies):
        plt.text(i - width/2, v + 2, f'{v:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(batch_latencies):
        plt.text(i + width/2, v + 2, f'{v:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    batch_per_sample = [latency / BATCH_SIZE for latency in batch_latencies]
    plt.figtext(0.5, 0.01, f'Average latency per sentence in batch mode: ' + 
               ', '.join([f'{model}: {latency:.2f}ms' for model, latency in zip(batch_names, batch_per_sample)]),
               ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()

def main():
    test_sentences = [
        "Hello, how are you?",
        "I would like to learn German.",
        "The weather is nice today.",
        "Where is the nearest restaurant?",
        "Can you help me with my homework?",
        "I enjoy listening to music in my free time.",
        "The quick brown fox jumps over the lazy dog.",
        "This is a complex sentence with multiple clauses and conjunctions.",
        "Artificial intelligence has made significant progress in recent years.",
        "Please translate this sentence from English to German."
    ]
    
    en_vocab_adapter, de_vocab_adapter = load_vocab()
    de_nlp = spacy.load('de_core_news_sm')
    
    input_dim = len(de_vocab_adapter)
    output_dim = len(en_vocab_adapter)
    
    print(f"Vocabulary sizes - German: {input_dim}, English: {output_dim}")
    
    pt_model_path = os.path.join(MODELS_DIR, "best.pt")
    onnx_encoder_path = os.path.join(MODELS_DIR, "encoder.onnx")
    onnx_decoder_path = os.path.join(MODELS_DIR, "decoder.onnx")
    quant_pt_model_path = os.path.join(MODELS_DIR, "best_quant.pt")
    quant_onnx_encoder_path = os.path.join(MODELS_DIR, "encoder_quant.onnx")
    quant_onnx_decoder_path = os.path.join(MODELS_DIR, "decoder_quant.onnx")
    
    results = []
    
    try:
        # Test PyTorch model - single inference
        torch_model = load_torch_model(pt_model_path, input_dim, output_dim)
        torch_results = benchmark_torch_model(torch_model, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "PyTorch")
        results.append(torch_results)
        
        # Test PyTorch model - batch inference
        torch_batch_results = benchmark_batch_torch_model(torch_model, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "PyTorch")
        results.append(torch_batch_results)
        
        del torch_model
        gc.collect()
    except Exception as e:
        print(f"Error benchmarking PyTorch model: {e}")
    
    try:
        if hasattr(torch.backends, 'quantized') and hasattr(torch.backends.quantized, 'supported_engines'):
            engines = torch.backends.quantized.supported_engines
            if engines:
                torch.backends.quantized.engine = engines[0]
        # Test Quantized PyTorch model - single inference
        quant_torch_model = load_torch_model(quant_pt_model_path, input_dim, output_dim)
        quant_torch_results = benchmark_torch_model(quant_torch_model, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "Quantized PyTorch")
        results.append(quant_torch_results)
        # Test Quantized PyTorch model - batch inference
        quant_torch_batch_results = benchmark_batch_torch_model(quant_torch_model, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "Quantized PyTorch")
        results.append(quant_torch_batch_results)
        
        del quant_torch_model
        gc.collect()
    except Exception as e:
        print(f"Error benchmarking Quantized PyTorch model: {e}")
        print("Creating freshly quantized model for benchmark...")
    
    try:
        # Test ONNX model - single inference
        encoder_sess, decoder_sess = load_onnx_sessions(onnx_encoder_path, onnx_decoder_path)
        onnx_results = benchmark_onnx_model(encoder_sess, decoder_sess, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "ONNX")
        results.append(onnx_results)
        
        # Test ONNX model - batch inference
        onnx_batch_results = benchmark_batch_onnx_model(encoder_sess, decoder_sess, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "ONNX")
        results.append(onnx_batch_results)
        
        del encoder_sess, decoder_sess
        gc.collect()
    except Exception as e:
        print(f"Error benchmarking ONNX model: {e}")
    
    try:
        # Test Quantized ONNX model - single inference
        quant_encoder_sess, quant_decoder_sess = load_onnx_sessions(quant_onnx_encoder_path, quant_onnx_decoder_path)
        quant_onnx_results = benchmark_onnx_model(quant_encoder_sess, quant_decoder_sess, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "Quantized ONNX")
        results.append(quant_onnx_results)
        
        # Test Quantized ONNX model - batch inference
        quant_onnx_batch_results = benchmark_batch_onnx_model(quant_encoder_sess, quant_decoder_sess, test_sentences, en_vocab_adapter, de_vocab_adapter, de_nlp, "Quantized ONNX")
        results.append(quant_onnx_batch_results)
        
        del quant_encoder_sess, quant_decoder_sess
        gc.collect()
    except Exception as e:
        print(f"Error benchmarking Quantized ONNX model: {e}")
    
    if not results:
        print("No benchmarks completed successfully. Check the errors above.")
        return
    
    print("\n===== BENCHMARK RESULTS =====")
    for result in results:
        print(f"{result['name']}:")
        print(f"  Average Latency: {result['latency']:.2f} ms")
        if 'prof_stats' in result:
            print(f"  Profiler Stats:")
            print(result['prof_stats'])
        print()
    
    batch_results = [result for result in results if "(Batch)" in result["name"]]
    if batch_results:
        print("\n===== PER-SAMPLE BATCH LATENCIES =====")
        for result in batch_results:
            per_sample_latency = result["latency"] / BATCH_SIZE
            print(f"{result['name']}:")
            print(f"  Per-sample Latency: {per_sample_latency:.2f} ms")
        print()
    
    plot_results(results)
    
    print("Benchmark complete. Results visualized in 'benchmark_results.png'")

if __name__ == "__main__":
    main()
