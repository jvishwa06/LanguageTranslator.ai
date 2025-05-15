import numpy as np
import torch
import spacy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import datasets
import tqdm
import json
import os
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
experiment_name = "german-english-translator"
mlflow.set_experiment(experiment_name)

os.makedirs("models", exist_ok=True)
os.makedirs("vocabs", exist_ok=True)

en_nlp = spacy.load('en_core_web_sm')
de_nlp = spacy.load('de_core_news_sm')

language_dataset = datasets.load_dataset('bentrevett/multi30k')
train_data, val_data, test_data = language_dataset['train'], language_dataset['validation'], language_dataset['test']

train_length = len(train_data)
val_length = len(val_data)
test_length = len(test_data)

print(f"Train data length: {train_length}")
print(f"Validation data length: {val_length}")
print(f"Test data length: {test_length}")

def tokenizer(sample, en_nlp, de_nlp, max_length, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(sample["en"])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(sample["de"])][:max_length]
    en_tokens = [token.lower() for token in en_tokens]
    de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}

fn_kwargs = {"en_nlp": en_nlp, "de_nlp": de_nlp, "max_length": 1000, "sos_token": '<sos>', "eos_token": '<eos>'}
train_data = train_data.map(tokenizer, fn_kwargs=fn_kwargs)
val_data = val_data.map(tokenizer, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenizer, fn_kwargs=fn_kwargs)

specials = ['<unk>', '<pad>', '<sos>', '<eos>']
en_vocab = build_vocab_from_iterator(train_data['en_tokens'], specials=specials)
de_vocab = build_vocab_from_iterator(train_data['de_tokens'], specials=specials)

unk_index = en_vocab['<unk>']
en_vocab.set_default_index(unk_index)
de_vocab.set_default_index(unk_index)

print(f"English vocabulary size: {len(en_vocab)}")
print(f"German vocabulary size: {len(de_vocab)}")

en_vocab_dict = {word: idx for idx, word in enumerate(en_vocab.get_itos())}
de_vocab_dict = {word: idx for idx, word in enumerate(de_vocab.get_itos())}

with open('vocabs/en_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(en_vocab_dict, f, ensure_ascii=False, indent=4)

with open('vocabs/de_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(de_vocab_dict, f, ensure_ascii=False, indent=4)

def numericalize(sample, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(sample["en_tokens"])
    de_ids = de_vocab.lookup_indices(sample["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}

fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}
train_data = train_data.map(numericalize, fn_kwargs=fn_kwargs)
val_data = val_data.map(numericalize, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize, fn_kwargs=fn_kwargs)

train_data = train_data.with_format(type="torch", columns=['en_ids', 'de_ids'], output_all_columns=True)
val_data = val_data.with_format(type="torch", columns=['en_ids', 'de_ids'], output_all_columns=True)
test_data = test_data.with_format(type="torch", columns=['en_ids', 'de_ids'], output_all_columns=True)

pad_index = en_vocab['<pad>']
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [sample["en_ids"] for sample in batch]
        batch_de_ids = [sample["de_ids"] for sample in batch]
        batch_en_ids = pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {"en_ids": batch_en_ids, "de_ids": batch_de_ids}
        return batch
    return collate_fn

def dataloader_func(dataset, batch_size, shuffle, pad_index):
    collate_fn = get_collate_fn(pad_index)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

batch_size = 512
train_loader = dataloader_func(train_data, batch_size=batch_size, shuffle=True, pad_index=pad_index)
val_loader = dataloader_func(val_data, batch_size=batch_size, shuffle=True, pad_index=pad_index)
test_loader = dataloader_func(test_data, batch_size=batch_size, shuffle=True, pad_index=pad_index)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        out, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
    
    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(0)
        emb = self.embedding(input_token)
        emb = self.dropout(emb)
        out, (hidden, cell) = self.lstm(emb, (hidden, cell))
        out = out.squeeze(0)
        pred = self.fc(out)
        return pred, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(self.device)
        input_token = trg[0, :]
        hidden, cell = self.encoder(src)
        for t in range(1, trg_len):
            out, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[t] = out
            top1 = out.argmax(1)
            teacher_force = np.random.random() < teacher_forcing_ratio
            input_token = trg[t] if teacher_force else top1
        return outputs

params = {
    "input_dim": len(de_vocab),
    "output_dim": len(en_vocab),
    "encoder_embedding_dim": 256,
    "decoder_embedding_dim": 256,
    "hidden_size": 512,
    "num_layers": 3,
    "encoder_dropout": 0.2,
    "decoder_dropout": 0.2,
    "batch_size": batch_size,
    "n_epochs": 20,
    "clip": 1.0,
    "teacher_forcing_ratio": 1.0,
    "learning_rate": 0.001
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

def train_func(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in data_loader:
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        pred = output.argmax(dim=-1)
        non_pad_mask = trg.ne(pad_index)
        correct = pred.eq(trg).masked_select(non_pad_mask).sum().item()
        correct_predictions += correct
        total_predictions += non_pad_mask.sum().item()

    accuracy = correct_predictions / total_predictions * 100
    avg_loss = epoch_loss / len(data_loader)
    perplexity = np.exp(avg_loss)

    return avg_loss, accuracy, perplexity

def evaluate_func(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            pred = output.argmax(dim=-1)

            non_pad_mask = trg.ne(pad_index)
            correct = pred.eq(trg).masked_select(non_pad_mask).sum().item()
            correct_predictions += correct
            total_predictions += non_pad_mask.sum().item()
    
    accuracy = correct_predictions / total_predictions * 100
    avg_loss = epoch_loss / len(data_loader)
    perplexity = np.exp(avg_loss)

    return avg_loss, accuracy, perplexity


def translate_func(sentence, model, de_nlp, en_vocab, de_vocab, sos_token, eos_token, device, max_output_length=25):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]

        tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = de_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens


with mlflow.start_run() as run:

    mlflow.log_params(params)

    mlflow.log_artifact("vocabs/en_vocab.json")
    mlflow.log_artifact("vocabs/de_vocab.json")
    
    encoder = Encoder(
        input_dim=params["input_dim"],
        embedding_dim=params["encoder_embedding_dim"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["encoder_dropout"]
    )

    decoder = Decoder(
        output_dim=params["output_dim"],
        embedding_dim=params["decoder_embedding_dim"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["decoder_dropout"]
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    
    mlflow.log_text(str(model), "model_architecture.txt")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    
    best_valid_loss = float("inf")
    
    for epoch in tqdm.tqdm(range(params["n_epochs"])):
        train_loss, train_accuracy, train_ppl = train_func(model, train_loader, optimizer, criterion, params["clip"], params["teacher_forcing_ratio"], device)
        
        valid_loss, valid_accuracy, valid_ppl = evaluate_func(model, val_loader, criterion, device)
        
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_perplexity": train_ppl,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "valid_perplexity": valid_ppl,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch+1}/{params['n_epochs']}:")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:.3f} | Train Acc: {train_accuracy:.2f}%")
        print(f"\tValid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:.3f} | Valid Acc: {valid_accuracy:.2f}%")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_path = "models/seq2seq_best.pt"
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(best_model_path)
            print(f"\tNew best model saved at epoch {epoch+1}")
    
    model.load_state_dict(torch.load("models/seq2seq_best.pt"))
    
    test_loss, test_accuracy, test_ppl = evaluate_func(model, test_loader, criterion, device)
    
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_perplexity": test_ppl
    })
    
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_accuracy:.2f}% | Test PPL: {test_ppl:.3f}")
    
    test_sentences = ["Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem","Ein Mann mit einem Hut, der etwas anstarrt."]
    
    translations = []
    for sentence in test_sentences:
        translation = translate_func(sentence, model, de_nlp, en_vocab, de_vocab, sos_token='<sos>', eos_token='<eos>', device=device)
        translation_text = ' '.join([t for t in translation if t not in ['<sos>', '<eos>']])
        translations.append(f"German: {sentence}\nEnglish: {translation_text}")
    
    mlflow.log_text("\n\n".join(translations), "example_translations.txt")
    
    mlflow.pytorch.log_model(model,"pytorch_model",registered_model_name="german_english_translator",)
    
    src = torch.randint(0, params["input_dim"], (10, 1)).to(device)
    torch.onnx.export(
        encoder, src, "models/encoder.onnx",
        input_names=["src"],
        output_names=["hidden", "cell"],
        dynamic_axes={"src": {0: "src_len"}},
        opset_version=16
    )
    
    input_token = torch.randint(0, params["output_dim"], (1,)).to(device)
    hidden = torch.randn(params["num_layers"] * 2, 1, params["hidden_size"]).to(device)
    cell = torch.randn(params["num_layers"] * 2, 1, params["hidden_size"]).to(device)
    
    torch.onnx.export(
        decoder, (input_token, hidden, cell), "models/decoder.onnx",
        input_names=["input_token", "hidden", "cell"],
        output_names=["output", "hidden_out", "cell_out"],
        dynamic_axes={
            "input_token": {0: "seq_len"},
            "hidden": {1: "batch_size"},
            "cell": {1: "batch_size"},
        },
        opset_version=16
    )
    
    mlflow.log_artifact("models/encoder.onnx")
    mlflow.log_artifact("models/decoder.onnx")
    
    print(f"MLflow run completed. Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")


def load_model_for_inference(run_id):
    artifact_path = f"mlruns/1/{run_id}/artifacts"
    
    with open(f"{artifact_path}/en_vocab.json", 'r', encoding='utf-8') as f:
        en_vocab_dict = json.load(f)
    
    with open(f"{artifact_path}/de_vocab.json", 'r', encoding='utf-8') as f:
        de_vocab_dict = json.load(f)
    
    from torchtext.vocab import vocab
    from collections import OrderedDict
    
    en_vocab = vocab(OrderedDict(sorted(en_vocab_dict.items(), key=lambda x: x[1])))
    de_vocab = vocab(OrderedDict(sorted(de_vocab_dict.items(), key=lambda x: x[1])))
    
    en_vocab.set_default_index(en_vocab['<unk>'])
    de_vocab.set_default_index(de_vocab['<unk>'])
    
    loaded_model = mlflow.pytorch.load_model(f"mlruns/1/{run_id}/artifacts/pytorch_model")
    
    return loaded_model, en_vocab, de_vocab

run_id = ""
model, en_vocab, de_vocab = load_model_for_inference(run_id)

sentence = "Ein Mann mit einem Hut, der etwas anstarrt."
translation = translate_func(sentence, model, de_nlp, en_vocab, de_vocab, sos_token='<sos>', eos_token='<eos>', device=device)
print("Translation:", ' '.join([t for t in translation if t not in ['<sos>', '<eos>']]))
