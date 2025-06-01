from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import uvicorn
import os
import json
import onnxruntime as ort
from contextlib import asynccontextmanager
from torchtext.vocab import build_vocab_from_iterator

from architecture import translate_onnx

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str
    original_text: str

max_len=25
encoder_sess = None; decoder_sess = None
en_nlp = None; de_nlp = None
en_vocab = None; de_vocab = None
sos_token = '<sos>'; eos_token = '<eos>'; pad_token = '<pad>'; unk_token = '<pad>'


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder_sess, decoder_sess, en_nlp, de_nlp, en_vocab, de_vocab
    
    try:
        print("Loading tokenizer models")
        en_nlp = spacy.load('en_core_web_sm')
        de_nlp = spacy.load('de_core_news_sm')
        
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        en_vocab_path = os.path.join(models_dir, "en_vocab.json")
        de_vocab_path = os.path.join(models_dir, "de_vocab.json")
        
        if os.path.exists(en_vocab_path) and os.path.exists(de_vocab_path):
            print("Loading saved vocabularies")
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
            
            print("Vocabularies loaded successfully")
        else:
            print("Error while loading Vocabularies")
        
        print("Loading quantized ONNX models")
        encoder_path = os.path.join(os.path.dirname(__file__), "models", "encoder_quant.onnx")
        decoder_path = os.path.join(os.path.dirname(__file__), "models", "decoder_quant.onnx")
        
        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            print("Quantized models not found, falling back to regular models...")
            encoder_path = os.path.join(os.path.dirname(__file__), "models", "encoder.onnx")
            decoder_path = os.path.join(os.path.dirname(__file__), "models", "decoder.onnx")
            
            if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
                raise FileNotFoundError(f"ONNX model files not found at {encoder_path} or {decoder_path}")
        
        providers = ['CPUExecutionProvider']
        
        encoder_sess = ort.InferenceSession(encoder_path, providers=providers)
        decoder_sess = ort.InferenceSession(decoder_path, providers=providers)
        
        print("ONNX models loaded successfully!")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise RuntimeError(f"Failed to load models and resources: {e}")
    
    yield
    print("Shutting down and cleaning up resources...")

app = FastAPI(
    title="German to English Translator (ONNX)", 
    description="API for translating German text to English using a Seq2Seq model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"message": "German to English Translation API (ONNX Runtime)", "status": "online"}

@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    try:
        translated_tokens = translate_onnx(request.text, de_nlp, en_vocab, de_vocab, encoder_sess, decoder_sess, sos_token, eos_token, max_len)
        
        filtered_tokens = [token for token in translated_tokens if token not in [sos_token, eos_token, pad_token, unk_token]]
        translated_text = " ".join(filtered_tokens)
        
        return TranslationResponse(original_text=request.text, translated_text=translated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)