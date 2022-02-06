import torch.cuda
from torch import nn

from modules.autoencoders import get_autoencoder, get_lstm_autoencoder
from modules.misc import LSTMDecoderEmpty, LSTMDecoderMixedTeacherForcing, DualLSTMDecoderAdapter


def get_ae_setup(setup_objective, ae_name, height, width, embedding_size, n_channels, ae_loss, n_layers,
                 decoder_input_type, teacher_forcing_probability, n_batches):
    if setup_objective == "RECONSTRUCTION" and "LSTM" in ae_name:
        return LSTMReconstructionSetup(setup_objective, ae_name, height, embedding_size, ae_loss, n_layers,
                                       decoder_input_type,
                                       teacher_forcing_probability, n_batches)
    elif setup_objective == "RECONSTRUCTION":
        return AutoencoderSetup(setup_objective, ae_name, height, width, embedding_size, n_channels, ae_loss)
    elif setup_objective == "PREDICTION":
        return LSTMPredictionSetup(setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                                   teacher_forcing_probability, n_batches)
    elif setup_objective == "HYBRID":
        return LSTMHybridSetup(setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                               teacher_forcing_probability, n_batches)
    else:
        print("Unknown autoencoder adapter name")
        raise NotImplementedError


class AutoencoderSetup(nn.Module):
    def __init__(self, setup_objective, ae_name, height, width, embedding_size, n_channels, ae_loss):
        super().__init__()
        self.objective = setup_objective
        self.name = ae_name
        self.height = height
        self.width = width
        self.embedding_size = embedding_size
        self.n_channels = n_channels
        self.ae_loss = ae_loss
        self.autoencoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        return self.autoencoder(x)

    def get_autoencoder_loss(self, prediction, original_input):
        return self.ae_loss(prediction, original_input)

    def set_new_autoencoder(self):
        self.autoencoder = get_autoencoder(
            self.name,
            self.height,
            self.width,
            self.embedding_size,
            self.n_channels,
        )
        self.autoencoder.to(self.device)

    def __str__(self):
        return f"Autoencoder_name: {self.name}, Autoencoder loss: {self.ae_loss}," \
               f" Embedding_size: {self.embedding_size}, Architecture:\n {self.autoencoder}"

    def __repr__(self):
        return f"{self.objective}_{self.name}"

    @staticmethod
    def step():
        pass


class LSTMAutoencoderSetup(nn.Module):
    def __init__(self, setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches):
        assert "LSTM" in ae_name.upper(), "Only LSTM autoencoders should be used in this setup"
        super().__init__()
        self.objective = setup_objective
        self.name = ae_name
        self.height = height
        self.embedding_size = embedding_size
        self.ae_loss = ae_loss
        self.decoder_input_type = decoder_input_type
        self.teacher_forcing_probability = teacher_forcing_probability
        self.n_layers = n_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autoencoder = None
        self.decoder_length = 15

    @staticmethod
    def add_ones_feature(batch_size, signal_length, x):
        ones_feature = torch.ones((batch_size, signal_length, 1))
        return torch.cat((x, ones_feature), dim=2)

    def get_encoder_length(self, total_length):
        return total_length - self.decoder_length

    def forward(self, x):
        raise NotImplementedError

    def get_autoencoder_loss(self, prediction, original_input):
        return self.ae_loss(prediction, original_input)

    def set_new_autoencoder(self):
        encoder = nn.LSTM(self.height + 1, self.height, batch_first=True, num_layers=self.n_layers, dropout=0.5)
        decoder = self.get_decoder()
        self.autoencoder = get_lstm_autoencoder(
            self.name,
            encoder,
            decoder,
            self.embedding_size,
        )
        self.autoencoder.to(self.device)

    def get_decoder(self):
        if self.decoder_input_type == "EMPTY":
            return LSTMDecoderEmpty(self.height + 1, self.height, batch_first=True, num_layers=self.n_layers,
                                    dropout=0.5)
        elif self.decoder_input_type == "RECURSIVE":
            return LSTMDecoderMixedTeacherForcing(self.height + 1, self.height, batch_first=True, num_layers=self.n_layers,
                                                  dropout=0.5, teacher_forcing_probability=0.0)
        elif self.decoder_input_type == "TEACHER_FORCING":
            return LSTMDecoderMixedTeacherForcing(self.height + 1, self.height, batch_first=True,
                                                  num_layers=self.n_layers,
                                                  dropout=0.5, teacher_forcing_probability=1.0)
        elif self.decoder_input_type == "MIXED_TEACHER_FORCING":
            return LSTMDecoderMixedTeacherForcing(self.height + 1, self.height, batch_first=True,
                                                  num_layers=self.n_layers,
                                                  teacher_forcing_probability=self.teacher_forcing_probability)
        else:
            valid_options = ["EMPTY", "RECURSIVE", "TEACHER_FORCING", "MIXED_TEACHER_FORCING"]
            print(f"Decoder input type was {self.decoder_input_type}, but must be in {valid_options}")
            raise

    @staticmethod
    def step():
        pass

    def __str__(self):
        base_str = f"Autoencoder_name: {self.name}, Autoencoder loss: {self.ae_loss}, Embedding_size: " \
                   f"{self.embedding_size}, n_layers: {self.n_layers}, decoder_input: {self.decoder_input_type}" \
                   f", Architecture:\n {self.autoencoder}, Objective: {self.objective}"
        if self.decoder_input_type == "MIXED_TEACHER_FORCING" or self.decoder_input_type == "DYNAMIC_TEACHER_FORCING":
            return f"{base_str}\nteacher_forcing_probability: {self.teacher_forcing_probability}"
        return base_str

    def __repr__(self):
        return f"{self.objective}_{self.name}"


class LSTMReconstructionSetup(LSTMAutoencoderSetup):
    def __init__(self, setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches):
        super().__init__(setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                         teacher_forcing_probability, n_batches)

    def forward(self, x):
        batch_size, _, height, signal_length = x.shape
        rnn_format = x.view(batch_size, height, signal_length).permute(0, 2, 1)
        lstm_input = self.add_ones_feature(batch_size, signal_length, rnn_format)
        # Encoder spans the entire length of the sequence
        encoder_input = lstm_input
        decoder_input = lstm_input.flip(1)[:, :self.decoder_length, :]
        embeds, prediction = self.autoencoder(encoder_input, decoder_input)
        return embeds, prediction

    def get_autoencoder_loss(self, reconstruction, original_input):
        # The last encoder output is at total_length
        reconstruction_truth = original_input.flip(3)[:, :, :, :self.decoder_length]
        return self.ae_loss(reconstruction, reconstruction_truth)


class LSTMPredictionSetup(LSTMAutoencoderSetup):
    def __init__(self, setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches):
        super().__init__(setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                         teacher_forcing_probability, n_batches)

    def forward(self, x):
        batch_size, _, height, signal_length = x.shape
        encoder_length = self.get_encoder_length(signal_length)
        rnn_format = x.view(batch_size, height, signal_length).permute(0, 2, 1)
        lstm_input = self.add_ones_feature(batch_size, signal_length, rnn_format)
        # Encoder spans the sequence_length - decoder_length
        encoder_input = lstm_input[:, :encoder_length, :]
        decoder_input = lstm_input[:, encoder_length:, :]
        embeds, prediction = self.autoencoder(encoder_input, decoder_input)
        return embeds, prediction

    def get_autoencoder_loss(self, prediction, original_input):
        # The last encoder output is at total_length - decoder_length
        encoder_length = self.get_encoder_length(original_input.shape[3])
        true_future = original_input[:, :, :, encoder_length:]
        return self.ae_loss(prediction, true_future)


class LSTMHybridSetup(LSTMAutoencoderSetup):
    def __init__(self, setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches, hyperparameter=1):
        super().__init__(setup_objective, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                         teacher_forcing_probability, n_batches)
        self.hyperparameter = hyperparameter

    def forward(self, x):
        batch_size, _, height, signal_length = x.shape
        encoder_length = self.get_encoder_length(signal_length)
        rnn_format = x.view(batch_size, height, signal_length).permute(0, 2, 1)
        lstm_input = self.add_ones_feature(batch_size, signal_length, rnn_format)
        # The encoder spans the sequence_length - decoder_length just like with the prediction setup
        encoder_part = lstm_input[:, :encoder_length, :]
        reconstruction_part = lstm_input.flip(1)[:, self.decoder_length:2*self.decoder_length, :]
        prediction_part = lstm_input[:, encoder_length:, :]
        decoder_part = torch.cat((reconstruction_part, prediction_part), dim=1)
        embeds, prediction_and_reconstruction = self.autoencoder(encoder_part, decoder_part)
        return embeds, prediction_and_reconstruction

    def get_autoencoder_loss(self, catted_decoder_output, original_input):
        # The last encoder input is at total_length - decoder_length
        encoder_length = self.get_encoder_length(original_input.shape[3])
        reconstruction_truth = original_input.flip(3)[:, :, :, self.decoder_length:2*self.decoder_length]
        prediction_truth = original_input[:, :, :, encoder_length:]
        reconstruction = catted_decoder_output[:, :, :, :self.decoder_length]
        prediction = catted_decoder_output[:, :, :, self.decoder_length:]
        reconstruction_loss = self.ae_loss(reconstruction, reconstruction_truth)
        prediction_loss = self.ae_loss(prediction, prediction_truth)
        return reconstruction_loss + self.hyperparameter * prediction_loss

    def set_new_autoencoder(self):
        encoder = nn.LSTM(self.height, self.height, batch_first=True, num_layers=self.n_layers, dropout=0.5)
        decoder1 = self.get_decoder()
        decoder2 = self.get_decoder()
        dual_decoder = DualLSTMDecoderAdapter(decoder1, decoder2)
        self.autoencoder = get_lstm_autoencoder(
            self.name,
            encoder,
            dual_decoder,
            self.embedding_size,
        )
        self.autoencoder.to(self.device)
