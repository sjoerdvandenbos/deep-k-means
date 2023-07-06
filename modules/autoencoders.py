import torch
from torch import nn

from modules.layers import OrthogonalLinearLayer, ReshapeLayer, InterpolateLayer
from modules.utils import get_post_conv_height_and_width, get_output_padding, get_output_padding_length, \
    get_post_conv_length
from modules.blocks import BottleneckConvBlock, BottleneckConvTransposeBlock, BasicConvBlock, BasicConvTransposeBlock, \
    ConvBlock1D, ConvTransposeBlock1D
from modules.inception import InceptionResNetV2


def get_autoencoder(name, height, width, embedding_size, n_channels,
                    n_layers=1, decoder_input_type="EMPTY") -> nn.Module:
    if name == "CONVO_AUTOENCODER":
        return ConvAutoencoder(height, width, embedding_size, n_channels)
    elif name == "CONVO_AUTOENCODER_1D":
        return ConvoAutoencoder1D(input_size=width, n_channels=height, embedding_size=embedding_size)
    elif name == "FC_AUTOENCODER":
        return StackedAutoencoder(height, width, embedding_size, n_channels)
    elif name == "OLM_AUTOENCODER":
        return OrthogonalSAE(height, width, embedding_size, n_channels)
    elif name == "RESNET_AUTOENCODER" or name == "RESNET_AUTOENCODER_50":
        return ResnetConvoAutoencoder(height, width, embedding_size, n_channels, BottleneckConvBlock,
                                      BottleneckConvTransposeBlock, (3, 4, 6, 3))
    elif name == "RESNET_AUTOENCODER_18":
        return ResnetConvoAutoencoder(height, width, embedding_size, n_channels, BasicConvBlock,
                                      BasicConvTransposeBlock, (2, 2, 2, 2))
    elif name == "RESNET_AUTOENCODER_34":
        return ResnetConvoAutoencoder(height, width, embedding_size, n_channels, BasicConvBlock,
                                      BasicConvTransposeBlock, (3, 4, 6, 3))
    elif name == "RESNET_AUTOENCODER_10":
        return ResnetConvoAutoencoder(height, width, embedding_size, n_channels, BasicConvBlock,
                                      BasicConvTransposeBlock, (1, 1, 1, 1))
    elif name == "VARIATIONAL_AUTOENCODER":
        return VariationalAutoencoder1d2(400)
    elif name == "INCEPTIONRESNETV2":
        return InceptionResNetV2(height, width, embedding_size, n_channels)
    else:
        print("Autoencoder not found!")
        exit()


def get_lstm_autoencoder(name, encoder, decoder, embedding_size):
    if name == "STACKED_LSTM_AUTOENCODER":
        return StackedLSTMAutoencoder(encoder, decoder, embedding_size)
    elif name == "LSTM_AUTOENCODER":
        return LSTMAutoencoder(encoder, decoder, embedding_size)
    else:
        print("Lstm autoencoder not found!")
        exit()


class LSTMAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, embedding_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        hidden_size = encoder.hidden_size
        n_layers = encoder.num_layers
        forced_embedding_size = 2 * hidden_size * n_layers
        assert embedding_size == forced_embedding_size, f"Invalid embedding size for this lstm_ae, required: " \
                                                        f"{forced_embedding_size} your input from cli: {embedding_size}"

    def forward(self, encoder_input, decoder_input):
        batch_size = encoder_input.shape[0] if self.encoder.batch_first else encoder_input.shape[1]
        _, (hidden, cell) = self.encoder.forward(encoder_input)
        embed_hidden = hidden.permute(1, 2, 0).reshape(batch_size, -1)
        embed_cell = cell.permute(1, 2, 0).reshape(batch_size, -1)
        embeddings = torch.cat((embed_hidden, embed_cell), dim=1)
        prepared = _prepare_decoder_input(encoder_input, decoder_input)
        x, _ = self.decoder(prepared, (hidden, cell))
        return embeddings, x.permute(0, 2, 1).unsqueeze(1).flip(-1)


class StackedLSTMAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, embedding_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.inner_encoder = nn.Linear(encoder.hidden_size * encoder.num_layers * 2, embedding_size)
        self.inner_decoder = nn.Linear(embedding_size, decoder.hidden_size * decoder.num_layers * 2)

    def forward(self, encoder_input, decoder_input):
        batch_size = encoder_input.shape[0] if self.encoder.batch_first else encoder_input.shape[1]
        _, (hidden, cell) = self.encoder.forward(encoder_input)
        # reshape hidden and cell to input for fully connected layer
        hidden = hidden.permute(1, 2, 0).reshape(batch_size, -1)
        cell = cell.permute(1, 2, 0).reshape(batch_size, -1)
        x = torch.cat((hidden, cell), dim=1)
        embeddings = self.inner_encoder(x)                                # Shape: [batch, embedding_size]
        x = self.inner_decoder(embeddings)
        # reshape to hidden and cell of the lstm decoder, the first dimension of x indexes the hidden/cell state
        x = x.view(batch_size, 2, self.encoder.num_layers, self.encoder.input_size).permute(1, 2, 0, 3)
        hidden = x[0, :, :, :].contiguous()
        cell = x[1, :, :, :].contiguous()
        prepared = _prepare_decoder_input(encoder_input, decoder_input)
        x, _ = self.decoder(prepared, (hidden, cell))
        return embeddings, x.permute(0, 2, 1).unsqueeze(1)


def _prepare_decoder_input(encoder_input, unprepared_decoder_input):
    last_output = encoder_input[:, -1, :].unsqueeze(1)
    to_be_used = unprepared_decoder_input[:, :-1, :]
    return torch.cat((last_output, to_be_used), dim=1)


class StackedAutoencoder(nn.Module):
    def __init__(self, height, width, embedding_size, n_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height*width*n_channels, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, embedding_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, height*width),
            ReshapeLayer(-1, n_channels, height, width)
        )

    def forward(self, x):
        x = self.forward_encoder(x)
        embeddings = x.clone()
        x = self.decoder(x)
        return embeddings, x

    def forward_encoder(self, x):
        return self.encoder(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, height, width, embedding_size, n_channels):
        super().__init__()
        padding1 = 2
        stride1 = 2
        kernel_size1 = 5
        post_conv_height_1, post_conv_width_1 = get_post_conv_height_and_width(
            height, width, stride1)
        output_height_padding_1, output_width_padding_1 = get_output_padding(height, width, stride1)

        padding2 = 2
        stride2 = 2
        kernel_size2 = 5
        post_conv_height_2, post_conv_width_2 = get_post_conv_height_and_width(
            post_conv_height_1, post_conv_width_1, stride2)
        output_height_padding_2, output_width_padding_2 = get_output_padding(post_conv_height_1, post_conv_width_1,
                                                                             stride2)

        padding3 = 1
        stride3 = 2
        kernel_size3 = 3
        post_conv_height_3, post_conv_width_3 = get_post_conv_height_and_width(
            post_conv_height_2, post_conv_width_2, stride3)
        output_height_padding_3, output_width_padding_3 = get_output_padding(post_conv_height_2, post_conv_width_2,
                                                                             stride3)

        linear_size = post_conv_height_3 * post_conv_width_3 * 128

        self.encoder = nn.ModuleList([
            nn.Conv2d(n_channels, 32, kernel_size=(kernel_size1, kernel_size1), stride=(stride1, stride1),
                      padding=(padding1, padding1)),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(kernel_size2, kernel_size2), stride=(stride2, stride2),
                      padding=(padding2, padding2)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(kernel_size3, kernel_size3), stride=(stride3, stride3),
                      padding=(padding3, padding3)),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_size, embedding_size),
        ])

        self.decoder = nn.ModuleList([
            nn.Linear(embedding_size, linear_size),
            ReshapeLayer(-1, 128, post_conv_height_3, post_conv_width_3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(kernel_size3, kernel_size3), stride=(stride3, stride3),
                               padding=(padding3, padding3),
                               output_padding=(output_height_padding_3, output_width_padding_3)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(kernel_size2, kernel_size2), stride=(stride2, stride2),
                               padding=(padding2, padding2),
                               output_padding=(output_height_padding_2, output_width_padding_2)),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_channels, kernel_size=(kernel_size1, kernel_size1), stride=(stride1, stride1),
                               padding=(padding1, padding1),
                               output_padding=(output_height_padding_1, output_width_padding_1)),
        ])

    def forward(self, x):
        x = self.forward_encoder(x)
        embedding = x
        for m in self.decoder:
            x = m(x)
        return embedding, x

    def forward_encoder(self, x):
        for m in self.encoder:
            x = m(x)
        return x


class OrthogonalSAE(nn.Module):
    def __init__(self, height, width, embedding_size, n_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            OrthogonalLinearLayer(height * width * n_channels, 500),
            nn.ReLU(),
            OrthogonalLinearLayer(500, 500),
            nn.ReLU(),
            OrthogonalLinearLayer(500, 2000),
            nn.ReLU(),
            OrthogonalLinearLayer(2000, embedding_size),
        )
        self.decoder = nn.Sequential(
            OrthogonalLinearLayer(embedding_size, 2000),
            nn.ReLU(),
            OrthogonalLinearLayer(2000, 500),
            nn.ReLU(),
            OrthogonalLinearLayer(500, 500),
            nn.ReLU(),
            OrthogonalLinearLayer(500, height * width),
            ReshapeLayer(-1, n_channels, height, width)
        )

    def forward(self, x):
        x = self.forward_encoder(x)
        embeddings = x
        x = self.decoder(x)
        return embeddings, x

    def forward_encoder(self, x):
        return self.encoder(x)


class ResnetConvoAutoencoder(nn.Module):
    def __init__(self, height, width, embedding_size, n_channels, blocktype, reverse_blocktype,
                 block_sections=(2, 2, 2, 2), base_n_channels=64):
        super().__init__()
        self.in_channels = base_n_channels

        kernel_size77 = (7, 7)
        padding77 = (3, 3)
        stride_step = (2, 2)
        kernel_size33 = (3, 3)
        padding33 = (1, 1)
        output_pad77 = get_output_padding(height, width, stride_step)

        height77, width77 = get_post_conv_height_and_width(height, width, stride_step)
        height_pool, width_pool = get_post_conv_height_and_width(height77, width77, stride_step)
        height128, width128 = get_post_conv_height_and_width(height_pool, width_pool, stride_step)
        height256, width256 = get_post_conv_height_and_width(height128, width128, stride_step)
        height512, width512 = get_post_conv_height_and_width(height256, width256, stride_step)

        self.maxpool_indices = None
        self.post_max_unpool_dims = height77, width77

        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(n_channels, self.in_channels, kernel_size77, stride=stride_step,
                                      padding=padding77, bias=False))
        self.encoder.append(nn.BatchNorm2d(self.in_channels))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Dropout2d(p=0.1)),
        self.encoder.append(nn.MaxPool2d(kernel_size33, stride_step, padding33, return_indices=True))

        self.encoder.append(self._make_section(blocktype, block_sections[0], n_channels=base_n_channels))
        self.encoder.append(self._make_section(blocktype, block_sections[1], n_channels=base_n_channels*2,
                                               stride=2))
        self.encoder.append(self._make_section(blocktype, block_sections[2], n_channels=base_n_channels*4,
                                               stride=2))
        self.encoder.append(self._make_section(blocktype, block_sections[3], n_channels=base_n_channels*8,
                                               stride=2))

        self.encoder.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.Linear(base_n_channels*8*blocktype.expansion, embedding_size))

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(embedding_size, base_n_channels*8*blocktype.expansion))
        self.decoder.append(ReshapeLayer(-1, base_n_channels*8*blocktype.expansion, 1, 1))
        self.decoder.append(InterpolateLayer((height512, width512)))

        self.decoder.append(self._make_reverse_section(height256, width256, reverse_blocktype, block_sections[3],
                                                       n_channels=base_n_channels*8, stride=2))
        self.decoder.append(self._make_reverse_section(height128, width128, reverse_blocktype, block_sections[2],
                                                       n_channels=base_n_channels*4, stride=2))
        self.decoder.append(self._make_reverse_section(height_pool, width_pool, reverse_blocktype, block_sections[1],
                                                       n_channels=base_n_channels*2, stride=2))
        self.decoder.append(self._make_reverse_section(height_pool, width_pool, reverse_blocktype, block_sections[0],
                                                       n_channels=base_n_channels, stride=1))
        self.decoder.append(nn.MaxUnpool2d(kernel_size33, stride_step, padding33))
        self.decoder.append(nn.ConvTranspose2d(self.in_channels, n_channels, kernel_size77, stride=stride_step,
                                               padding=padding77, output_padding=output_pad77, bias=True))

    def _make_section(self, blocktype, n_blocks, n_channels, stride=1):
        layers = []
        down_sampler = None
        # Create down sampler
        if stride != 1 or self.in_channels != n_channels*blocktype.expansion:
            down_sampler = nn.Sequential(
                nn.Conv2d(self.in_channels, n_channels*blocktype.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(n_channels*blocktype.expansion),
                # nn.Dropout2d(p=0.1),
            )
        # Create first block potentially changing the number of channels
        layers.append(blocktype(self.in_channels, n_channels, down_sampler=down_sampler, stride=stride))
        self.in_channels = n_channels*blocktype.expansion
        # Create rest of blocks with constant number of channels
        for _ in range(n_blocks - 1):
            layers.append(blocktype(self.in_channels, n_channels))
        return nn.Sequential(*layers)

    def _make_reverse_section(self, post_conv_transpose_height, post_conv_transpose_width, blocktype, n_blocks,
                              n_channels, stride=1):
        post_conv_height, post_conv_width = get_post_conv_height_and_width(post_conv_transpose_height,
                                                                           post_conv_transpose_width, 2)
        output_pad_height, output_pad_width = get_output_padding(post_conv_transpose_height,
                                                                 post_conv_transpose_width, stride)
        layers = []
        up_sampler = None
        # Create blocks with constant number of channels
        for _ in range(n_blocks - 1):
            layers.append(blocktype(post_conv_height, post_conv_width, self.in_channels, n_channels))
        if blocktype == BottleneckConvTransposeBlock:
            # Create up sampler
            up_sampler = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, n_channels*stride, kernel_size=1, stride=stride,
                                   padding=0, output_padding=(output_pad_height, output_pad_width)),
                nn.BatchNorm2d(n_channels*stride),
                # nn.Dropout2d(p=0.1),
            )
            # Create last block potentially changing the number of channels
            layers.append(blocktype(post_conv_transpose_height, post_conv_transpose_width, self.in_channels,
                                    n_channels, up_sampler=up_sampler, stride=stride, last=True))
            self.in_channels = n_channels*stride
        else:
            # Create up sampler
            up_sampler = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, n_channels//stride, kernel_size=1, stride=stride,
                                   padding=0, output_padding=(output_pad_height, output_pad_width)),
                nn.BatchNorm2d(n_channels//stride),
                # nn.Dropout2d(p=0.1),
            )
            # Create last block potentially changing the number of channels
            layers.append(blocktype(post_conv_transpose_height, post_conv_transpose_width, self.in_channels,
                                    n_channels//stride, up_sampler=up_sampler, stride=stride, last=True))
            self.in_channels = n_channels//stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_encoder(x)
        embedding = x
        for module in self.decoder:
            x = self._forward_single_module(module, x)  # Decoder forward
        return embedding, x

    def forward_encoder(self, x):
        for module in self.encoder:
            x = self._forward_single_module(module, x)  # Encoder forward
        return x

    def _forward_single_module(self, module, x):
        if type(module) == nn.MaxPool2d:
            x, indices = module(x)
            self.maxpool_indices = indices
            return x
        elif type(module) == nn.MaxUnpool2d:
            return module(x, self.maxpool_indices, output_size=self.post_max_unpool_dims)
        else:
            return module(x)


class VariationalAutoencoder1d(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = input_size // 16
        self.conv_encoder = nn.Sequential(
            ConvBlock1D(in_channels=1, out_channels=8, kernel_size=11, padding=5, stride=1),
            nn.MaxPool1d(2, 2, 0, return_indices=True),
            ConvBlock1D(in_channels=8, out_channels=16, kernel_size=7, padding=3, stride=1),
            nn.MaxPool1d(2, 2, 0, return_indices=True),
            ConvBlock1D(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.MaxPool1d(2, 2, 0, return_indices=True),
            ConvBlock1D(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool1d(2, 2, 0, return_indices=True),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )
        output_pad1 = get_output_padding_length(50, 2)
        output_pad2 = get_output_padding_length(100, 2)
        output_pad3 = get_output_padding_length(200, 2)
        output_pad4 = get_output_padding_length(400, 2)
        self.conv_decoder = nn.Sequential(
            ReshapeLayer(-1, 1, self.embedding_size),
            nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1,
                               output_padding=output_pad1),
            # nn.MaxUnpool1d(2, 2, 0),
            ConvTransposeBlock1D(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,
                                 output_padding=output_pad2),
            # nn.MaxUnpool1d(2, 2, 0),
            ConvTransposeBlock1D(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1,
                                 output_padding=output_pad3),
            # nn.MaxUnpool1d(2, 2, 0),
            ConvTransposeBlock1D(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1,
                                 output_padding=output_pad4),
            # nn.MaxUnpool1d(2, 2, 0),
            ConvTransposeBlock1D(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0),
        )
        self.linear_encoder = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.Linear(input_size//2, input_size//8),
            nn.Linear(input_size//8, self.embedding_size),
        )
        self.linear_decoder = nn.Sequential(
            nn.Linear(self.embedding_size, input_size//8),
            nn.Linear(input_size//8, input_size//2),
            nn.Linear(input_size//2, input_size),
        )
        self.mean_layer = nn.Linear(2*self.embedding_size, self.embedding_size)
        self.std_dev_layer = nn.Linear(2*self.embedding_size, self.embedding_size)
        self.decoding_combiner = nn.Linear(2*input_size, input_size)

    def forward(self, x):
        x = x.squeeze(2)
        conv_encoding = self.conv_encoder(x)
        linear_encoding = self.linear_encoder(x.squeeze(1))
        catted_encoding = torch.cat((conv_encoding, linear_encoding), dim=1)

        means = self.mean_layer(catted_encoding)
        log_var = self.std_dev_layer(catted_encoding)
        standard_samples = torch.randn_like(means, device="cuda")
        embeds = means + standard_samples * log_var.exp().sqrt()

        linear_decoding = self.linear_decoder(embeds)
        conv_decoding = self.conv_decoder(embeds)
        catted_decoding = torch.cat((linear_decoding, conv_decoding.squeeze(1)), dim=1)
        reconstruction = self.decoding_combiner(catted_decoding)
        return embeds, reconstruction.unsqueeze(1).unsqueeze(2), means, log_var

    def decode(self, embeds):
        linear_decoding = self.linear_decoder(embeds)
        conv_decoding = self.conv_decoder(embeds.unsqueeze(1))
        catted_decoding = torch.cat((linear_decoding, conv_decoding.squeeze(1)), dim=1)
        reconstruction = self.decoding_combiner(catted_decoding)
        return reconstruction


class VariationalAutoencoder1d2(VariationalAutoencoder1d):
    def __init__(self, input_size):
        super().__init__(input_size)
        self.input_size = input_size
        self.embedding_size = input_size // 16

        self.conv_encoder = nn.Sequential(
            ReshapeLayer(-1, 1, input_size),
            ConvBlock1D(1, 8, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock1D(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock1D(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock1D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )

        self.conv_decoder = nn.Sequential(
            ReshapeLayer(-1, 1, self.embedding_size),
            ConvBlock1D(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            ConvBlock1D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            ConvBlock1D(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            ConvBlock1D(16, 8, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
        )


class ConvoAutoencoder1D(nn.Module):
    def __init__(self, input_size, n_channels, embedding_size):
        super().__init__()
        post1_size = get_post_conv_length(input_size, stride=2, kernel_size=7, padding=0)
        post2_size = get_post_conv_length(post1_size, stride=2, kernel_size=5, padding=0)
        post3_size = get_post_conv_length(post2_size, stride=2, kernel_size=3, padding=0)
        output_padding1 = get_output_padding_length(input_size, stride=2)
        output_padding2 = get_output_padding_length(post1_size, stride=2)
        output_padding3 = get_output_padding_length(post2_size, stride=2)
        self.encoder = nn.Sequential(
            ConvBlock1D(n_channels, 32, kernel_size=7, stride=2, padding=0),
            ConvBlock1D(32, 64, kernel_size=5, stride=2, padding=0),
            ConvBlock1D(64, 128, kernel_size=3, stride=2, padding=0, activation=False),
            nn.Flatten(),
            nn.Linear(post3_size*128, embedding_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, post3_size*128),
            ReshapeLayer(-1, 128, post3_size),
            ConvTransposeBlock1D(128, 64, kernel_size=3, stride=2, padding=0, output_padding=output_padding3),
            ConvTransposeBlock1D(64, 32, kernel_size=5, stride=2, padding=0, output_padding=output_padding2),
            ConvTransposeBlock1D(32, n_channels, kernel_size=7, stride=2, padding=0, output_padding=output_padding1,
                                 batch_norm=False, activation=False),
        )

    def forward(self, x):
        x = self.encoder(x)
        embeds = x
        x = self.decoder(x)
        return embeds, x


if __name__ == "__main__":
    iput = torch.rand(1, 3, 1000)
    _, c, s = iput.shape
    encoder = ConvoAutoencoder1D(s, c, 10)
    e, r = encoder(iput)
    print(r.shape)
