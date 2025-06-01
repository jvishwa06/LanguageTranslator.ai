import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_size,num_layers,dropout):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_size,num_layers=num_layers,bidirectional=True,dropout=dropout)
    def forward(self,src):
        embedded = self.dropout(self.embedding(src))
        _,(hidden,cell) = self.lstm(embedded)
        return hidden,cell

class Decoder(nn.Module):
    def __init__(self,output_dim,embedding_dim,hidden_size,num_layers,dropout):
        super(Decoder,self).__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_dim,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_size,num_layers=num_layers,dropout=dropout)
        self.fc = nn.Linear(hidden_size*2,output_dim)
    def forward(self,input_token,hidden,cell):
        input_token = input_token.unsqueeze(0)
        emb = self.embedding(input_token)
        emb = self.dropout(emb)
        out,(hidden,cell) = self.lstm(emb,(hidden,cell))
        out = out.squeeze(0)
        pred = self.fc(out)
        return pred,hidden,cell

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self,src,trg,teacher_forcing_ratio):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len,batch_size,vocab_size).to(self.device)
        input_token = trg[0,:]
        hidden,cell = self.encoder(src)
        for t in range(1,trg_len):
            out,hidden,cell = self.decoder(input_token,hidden,cell)
            outputs[t] = out
            top1 = out.argmax(1)
            teacher_force = np.random.randn()<teacher_forcing_ratio
            input_token = trg[t] if teacher_force else top1
        return outputs

def translate_torch(sentence, model, de_nlp, en_vocab, de_vocab, sos_token, eos_token, device, max_len=25):
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
        for _ in range(max_len):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens

def translate_onnx(sentence, de_nlp, en_vocab, de_vocab, encoder_sess, decoder_sess, sos_token, eos_token, max_len=25):
    tokens = [sos_token] + [tok.text.lower() for tok in de_nlp(sentence)] + [eos_token]
    input_ids = de_vocab.lookup_indices(tokens)
    src_tensor = np.array(input_ids, dtype=np.int64).reshape(-1, 1) 

    encoder_outputs = encoder_sess.run(None, {"src": src_tensor})
    hidden, cell = encoder_outputs

    inputs = [en_vocab[sos_token]]
    translated_tokens = [sos_token]

    for _ in range(max_len):
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
        translated_tokens.append(en_vocab.lookup_token(pred_token_id))
        if pred_token_id == en_vocab[eos_token]:
            break

    return translated_tokens