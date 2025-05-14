from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import spacy
import uvicorn
import os
import datasets
import json
from contextlib import asynccontextmanager
from torchtext.vocab import build_vocab_from_iterator

from architecture import Encoder, Decoder, Seq2Seq, translate_torch

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str
    original_text: str

device = None; max_len = 25
encoder_sess = None; decoder_sess = None
en_nlp = None; de_nlp = None
en_vocab = None; de_vocab = None
sos_token = '<sos>'; eos_token = '<eos>'; pad_token = '<pad>'; unk_token = '<pad>'

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, en_nlp, de_nlp, en_vocab, de_vocab, device
    
    try:
        print("Loading spaCy models...")
        en_nlp = spacy.load('en_core_web_sm')
        de_nlp = spacy.load('de_core_news_sm')
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        en_vocab_path = os.path.join(models_dir, "en_vocab.json")
        de_vocab_path = os.path.join(models_dir, "de_vocab.json")
        
        if os.path.exists(en_vocab_path) and os.path.exists(de_vocab_path):
            print("Loading vocabularies from JSON files...")
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
            
            print("Vocabularies loaded successfully!")
        else:
            print("Building vocabularies from scratch...")
            print("Loading dataset...")
            language_dataset = datasets.load_dataset('bentrevett/multi30k')
            train_data = language_dataset['train']
            
            print("Tokenizing dataset...")
            def get_tokens(sample):
                en_tokens = [token.text for token in en_nlp.tokenizer(sample["en"])]
                de_tokens = [token.text for token in de_nlp.tokenizer(sample["de"])]
                en_tokens = [token.lower() for token in en_tokens]
                de_tokens = [token.lower() for token in de_tokens]
                en_tokens = [sos_token] + en_tokens + [eos_token]
                de_tokens = [sos_token] + de_tokens + [eos_token]
                return {"en_tokens": en_tokens, "de_tokens": de_tokens}
            
            tokenized_data = train_data.map(lambda x: get_tokens(x), load_from_cache_file=True)
            
            print("Building vocabularies...")
            specials = ['<unk>', '<pad>', '<sos>', '<eos>']
            en_vocab = build_vocab_from_iterator(tokenized_data['en_tokens'], specials=specials)
            de_vocab = build_vocab_from_iterator(tokenized_data['de_tokens'], specials=specials)
            
            unk_index = en_vocab['<unk>']
            en_vocab.set_default_index(unk_index)
            de_vocab.set_default_index(unk_index)
            
            print("Saving vocabularies to JSON files...")
            en_vocab_dict = {token: en_vocab[token] for token in en_vocab.get_itos()}
            de_vocab_dict = {token: de_vocab[token] for token in de_vocab.get_itos()}
            
            with open(en_vocab_path, 'w', encoding='utf-8') as f:
                json.dump(en_vocab_dict, f, ensure_ascii=False, indent=2)
            with open(de_vocab_path, 'w', encoding='utf-8') as f:
                json.dump(de_vocab_dict, f, ensure_ascii=False, indent=2)
                
            print("Vocabularies saved successfully!")
        
        print("Loading model directly...")
        input_dim = len(de_vocab)
        output_dim = len(en_vocab)
        
        model_path = os.path.join(os.path.dirname(__file__), "models", "seq2seq47.pt")
        
        encoder = Encoder(input_dim=input_dim, embedding_dim=256, hidden_size=512, num_layers=3, dropout=0.2)
        
        decoder = Decoder(output_dim=output_dim, embedding_dim=256, hidden_size=512, num_layers=3, dropout=0.2)
        
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        print("Loading model weights...")
        model_path = os.path.join(os.path.dirname(__file__), "models", "seq2seq47.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print("Model and resources loaded successfully!")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise RuntimeError(f"Failed to load model and resources: {e}")
    
    yield
    print("Shutting down and cleaning up resources...")

app = FastAPI(
    title="German to English Translator", 
    description="API for translating German text to English using a Seq2Seq model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"message": "German to English Translation API", "status": "online"}

@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        translated_tokens = translate_torch(request.text, model, en_nlp, de_nlp, en_vocab, de_vocab, sos_token, eos_token, device, max_len)
        
        filtered_tokens = [token for token in translated_tokens if token not in [sos_token, eos_token, pad_token, unk_token]]
        translated_text = " ".join(filtered_tokens)
        
        return TranslationResponse(original_text=request.text, translated_text=translated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("maintorch:app", host="0.0.0.0", port=8000, reload=True)
