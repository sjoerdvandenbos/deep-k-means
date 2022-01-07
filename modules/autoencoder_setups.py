from torch import nn

from modules.autoencoders import get_autoencoder, get_lstm_autoencoder
from modules.misc import LSTMDecoderEmpty, LSTMDecoderRecursive, LSTMDecoderMixedTeacherForcing, DualLSTMDecoderAdapter


def get_ae_setup(setup_objective, ae_name, height, width, embedding_size, n_channels, ae_loss, n_layers,
                 decoder_input_type, teacher_forcing_probability, n_batches):
    if setup_objective == "RECONSTRUCTION" and "LSTM" in ae_name:
        return LSTMReconstructionSetup(ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                                       teacher_forcing_probability, n_batches)
    elif setup_objective == "RECONSTRUCTION":
        return AutoencoderSetup(ae_name, height, width, embedding_size, n_channels, ae_loss)
    elif setup_objective == "PREDICTION":
        return LSTMPredictionSetup(ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                                   teacher_forcing_probability, n_batches)
    elif setup_objective == "HYBRID":
        return LSTMHybridSetup(ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                               teacher_forcing_probability, n_batches)
    else:
        print("Unknown autoencoder adapter name")
        raise NotImplementedError


class AutoencoderSetup(nn.Module):
    def __init__(self, ae_name, height, width, embedding_size, n_channels, ae_loss):
        super().__init__()
        self.name = ae_name
        self.height = height
        self.width = width
        self.embedding_size = embedding_size
        self.n_channels = n_channels
        self.ae_loss = ae_loss
        self.autoencoder = None

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

    def __str__(self):
        return f"Autoencoder_name: {self.name}, Autoencoder loss: {self.ae_loss}, Embedding_size: {self.embedding_size}"

    @staticmethod
    def step():
        pass


class LSTMAutoencoderSetup(nn.Module):
    def __init__(self, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches):
        assert "LSTM" in ae_name.upper(), "Only LSTM autoencoders should be used in this setup"
        super().__init__()
        self.name = ae_name
        self.height = height
        self.embedding_size = embedding_size
        self.ae_loss = ae_loss
        self.decoder_input_type = decoder_input_type
        self.teacher_forcing_probability = teacher_forcing_probability
        self.n_layers = n_layers
        self.autoencoder = None

    def forward(self, x):
        raise NotImplementedError

    def get_autoencoder_loss(self, prediction, original_input):
        return self.ae_loss(prediction, original_input)

    @staticmethod
    def get_prediction_boundary(x):
        length = x.shape[3]
        return length // 2

    def set_new_autoencoder(self):
        encoder = nn.LSTM(self.height, self.height, batch_first=True, num_layers=self.n_layers, dropout=0.5)
        decoder = self.get_decoder()
        self.autoencoder = get_lstm_autoencoder(
            self.name,
            encoder,
            decoder,
            self.embedding_size,
        )

    def get_decoder(self):
        if self.decoder_input_type == "EMPTY":
            return LSTMDecoderEmpty(self.height, self.height, batch_first=True, num_layers=self.n_layers,
                                    dropout=0.5)
        elif self.decoder_input_type == "RECURSIVE":
            return LSTMDecoderRecursive(self.height, self.height, batch_first=True, num_layers=self.n_layers,
                                        dropout=0.5)
        elif self.decoder_input_type == "TEACHER_FORCING":
            return LSTMDecoderMixedTeacherForcing(self.height, self.height, batch_first=True,
                                                  num_layers=self.n_layers,
                                                  dropout=0.5, teacher_forcing_probability=1.0)
        elif self.decoder_input_type == "MIXED_TEACHER_FORCING":
            return LSTMDecoderMixedTeacherForcing(self.height, self.height, batch_first=True,
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
        return f"Autoencoder_name: {self.name}, Autoencoder loss: {self.ae_loss}, Embedding_size: {self.embedding_size}"


class LSTMPredictionSetup(LSTMAutoencoderSetup):
    def __init__(self, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches):
        super().__init__(ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                         teacher_forcing_probability, n_batches)

    def forward(self, x):
        batch_size, _, height, signal_length = x.shape
        lstm_input = x.view(batch_size, height, signal_length).permute(0, 2, 1)
        encoder_input, decoder_input = lstm_input.split(2, dim=1)
        embeds, prediction = self.autoencoder(encoder_input, decoder_input)
        return embeds, prediction

    def get_autoencoder_loss(self, prediction, original_input):
        _, true_future = original_input.split(2, dim=3)
        return self.ae_loss(prediction, true_future)


class LSTMReconstructionSetup(LSTMAutoencoderSetup):
    def __init__(self, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches):
        super().__init__(ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                         teacher_forcing_probability, n_batches)

    def forward(self, x):
        batch_size, _, height, signal_length = x.shape
        lstm_input = x.view(batch_size, height, signal_length).permute(0, 2, 1)
        embeds, prediction = self.autoencoder(lstm_input, lstm_input.flip(1))
        return embeds, prediction

    def get_autoencoder_loss(self, reconstruction, original_input):
        return self.ae_loss(reconstruction, original_input.flip(3))


class LSTMHybridSetup(LSTMAutoencoderSetup):
    def __init__(self, ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                 teacher_forcing_probability, n_batches, hyperparameter=1):
        super().__init__(ae_name, height, embedding_size, ae_loss, n_layers, decoder_input_type,
                         teacher_forcing_probability, n_batches)
        self.hyperparameter = hyperparameter

    def forward(self, x):
        batch_size, _, height, signal_length = x.shape
        lstm_input = x.view(batch_size, height, signal_length).permute(0, 2, 1)
        embeds, prediction_and_reconstruction = self.autoencoder(lstm_input, lstm_input)
        return embeds, prediction_and_reconstruction

    def get_autoencoder_loss(self, catted_decoder_output, original_input):
        reconstruction_truth, prediction_truth = original_input.split(2, dim=3)
        reconstruction, prediction = catted_decoder_output.split(2, dim=3)
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
