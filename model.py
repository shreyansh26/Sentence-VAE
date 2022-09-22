import torch
from torch import nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, bidirectional=False, model_type='gru'):
        super().__init__()
        
        if model_type == 'gru':
            self.text_encoder = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.text_encoder = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, inp):
        return self.text_encoder(inp)

class TextDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, bidirectional=False, model_type='gru'):
        super().__init__()
        
        if model_type == 'gru':
            self.text_decoder = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.text_decoder = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, inp, hidden):
        return self.text_decoder(inp, hidden)

class SentenceVAEModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, word_dropout_rate, embedding_dropout_rate, 
                latent_dim, special_tokens, max_seq_length, model_type='gru', bidirectional=False, num_layers=1):
        
        super().__init__()

        self.hidden_size = hidden_size
        self.word_dropout_rate = word_dropout_rate
        self.latent_dim = latent_dim
        self.special_tokens = special_tokens
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_rate)

        self.text_encoder = TextEncoder(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.text_decoder = TextDecoder(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(self.device)

        if bidirectional:
            self.hidden_size_factor = 2
        else:
            self.hidden_size_factor = 1

        self.encoder_to_latent_mu = nn.Linear(self.hidden_size_factor * num_layers * hidden_size, latent_dim)
        self.encoder_to_latent_logsigma = nn.Linear(self.hidden_size_factor * num_layers * hidden_size, latent_dim)
        self.latent_to_decoder = nn.Linear(latent_dim, self.hidden_size_factor * num_layers * hidden_size)
        self.output_to_vocab = nn.Linear(self.hidden_size_factor * hidden_size, vocab_size)


    def forward(self, input_seq, lengths):
        batch_size = input_seq.shape[0]

        inp_embedding = self.embedding_layer(input_seq)

        # Encoder pass
        padded_input = nn.utils.rnn.pack_padded_sequence(inp_embedding, lengths, batch_first=True, enforce_sorted=False)

        enc_out, enc_hidden = self.text_encoder(padded_input)

        enc_hidden = enc_hidden.view(batch_size, self.hidden_size_factor * self.num_layers * self.hidden_size)

        # Reparametrization Trick
        mu = self.encoder_to_latent_mu(enc_hidden)
        log_sigmasquared = self.encoder_to_latent_logsigma(enc_hidden)
        sigma = torch.exp(0.5 * log_sigmasquared)
        z = mu + sigma * self.N.sample(mu.shape)
        
        # Decoder pass
        dec_hidden = self.latent_to_decoder(z)

        dec_hidden = dec_hidden.view(self.hidden_size_factor * self.num_layers, batch_size, self.hidden_size)
        
        if self.word_dropout_rate > 0:
            # Randomly replace decoder input with <unk>
            prob = torch.rand(input_seq.shape).to(self.device)

            decoder_inp_seq = input_seq.clone()

            prob[decoder_inp_seq.data == self.special_tokens.get('sos_token')] = 1
            prob[decoder_inp_seq.data == self.special_tokens.get('pad_token')] = 1

            decoder_inp_seq[prob < self.word_dropout_rate] = self.special_tokens.get('unk_token')
            inp_embedding = self.embedding_layer(decoder_inp_seq)

        inp_embedding = self.embedding_dropout(inp_embedding)
        padded_input = nn.utils.rnn.pack_padded_sequence(inp_embedding, lengths, batch_first=True, enforce_sorted=False)

        dec_out, dec_hidden = self.text_decoder(padded_input, dec_hidden)

        # Unpack
        padded_output, padded_len = nn.utils.rnn.pad_packed_sequence(dec_out, batch_first=True)
        # padded_output = padded_output.contiguous()
        b,s,_ = padded_output.shape

        # Project outputs to vocab
        padded_output = padded_output.view(-1, padded_output.size(2))
        logits = self.output_to_vocab(padded_output)
        logp = F.log_softmax(logits, dim=-1) # Softmax along the columns

        logp = logp.view(b, s, self.embedding_layer.num_embeddings)

        return logp, mu, log_sigmasquared, z

    def inference(self, n=4, z=None):
        '''
        Implements historyless decoding based on latent vector
        '''
        if z is None:
            batch_size = n
            z = torch.randn([batch_size, self.latent_dim]).to(self.device)
        else:
            batch_size = z.shape[0]
        
        hidden = self.latent_to_decoder(z)

        hidden = hidden.view(self.hidden_size_factor * self.num_layers, batch_size, self.hidden_size)

        generations = torch.Tensor(batch_size, self.max_seq_length).fill_(self.special_tokens.get('pad_token')).long()

        for idx in range(batch_size):
            curr_hidden = hidden[:, idx, :].unsqueeze(1).contiguous()

            cur_len = 0

            while cur_len < self.max_seq_length:
                if cur_len == 0:
                    input_sequence = torch.tensor([self.special_tokens.get('sos_token')]).to(self.device)
                
                generations[idx, cur_len] = input_sequence.data

                if input_sequence.item() == self.special_tokens.get('eos_token'):
                    break

                input_sequence = input_sequence.unsqueeze(1)

                input_embedding = self.embedding_layer(input_sequence)

                # Historyless decoding
                dec_output, curr_hidden = self.text_decoder(input_embedding, curr_hidden)
                # Generate logits over vocab size
                logits = self.output_to_vocab(dec_output)
                # Mask <unk> token so that is not in the output, take next best
                logits[:,:,self.special_tokens.get('unk_token')] = -100
                # Greedy decoding - take next best
                input_sequence = self._sample(logits)

                cur_len += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample
        