import argparse
import os
from math import ceil

from unsupervised_learning import UnsupervisedLearner
from supervised_learning import SupervisedLearner
from modules.autoencoder_setups import get_ae_setup
from utils import path_contains_dataset

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
    parser = argparse.ArgumentParser(description="Deep k-means algorithm")
    parser.add_argument("-d", "--dataset", type=str.upper,
                        help="Dataset on which to run (one of PTB, PTBMAT, MNIST, MIT, CIFAR10)", required=True)
    parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
    parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true')
    parser.add_argument("-l", "--lambda", type=float, default=0.1, dest="lambda_",
                        help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")
    parser.add_argument("-e", "--p_epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("-f", "--f_epochs", type=int, default=100, help="Number of fine-tuning epochs per alpha value")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="Size of the minibatches used by the optimizer")
    parser.add_argument("-n", "--number_runs", type=int, default=1,
                        help="number of repetitions of the entire experiment")
    parser.add_argument("-a", "--autoencoder", type=str.upper, default="STACKED_LSTM_AUTOENCODER",
                        choices=["FC_AUTOENCODER", "OLM_AUTOENCODER", "RESNET_AUTOENCODER",
                                 "RESNET_AUTOENCODER_18", "RESNET_AUTOENCODER_34",
                                 "RESNET_AUTOENCODER_50", "CONVO_AUTOENCODER", "RESNET_AUTOENCODER10",
                                 "LSTM_AUTOENCODER", "STACKED_LSTM_AUTOENCODER"],
                        nargs="?", help="type of autoencoder to use")
    parser.add_argument("-w", "--write_files", default=False, action="store_true",
                        help="if enabled, will write files to disk")
    parser.add_argument("-o", "--loss", type=str, default="MSE", help="type of loss function")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--ae_objective", type=str.upper, default="RECONSTRUCTION",
                        choices=["RECONSTRUCTION", "PREDICTION", "HYBRID"], nargs="?")
    parser.add_argument("--decoder_input", type=str.upper, default="RECURSIVE",
                        choices=["EMPTY", "RECURSIVE", "TEACHER_FORCING", "MIXED_TEACHER_FORCING"], nargs="?")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--teacher_forcing_probability", type=float, default=-1.)
    parser.add_argument("--polar_mapping", action="store_true")
    parser.add_argument("--embedding_size", type=int, default=2)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    if not path_contains_dataset(args.dataset_path):
        exit()

    # Dataset setting from arguments
    if args.dataset == "MNIST":
        from datasets_specs import mnist_specs as specs
    elif args.dataset == "PTB" or args.dataset == "PTBMAT":
        from datasets_specs.PTB import PTBSpecs
        specs = PTBSpecs(args.dataset_path)
    elif args.dataset == "MIT":
        from datasets_specs import mit_specs as specs
    elif args.dataset == "CIFAR10":
        from datasets_specs import cifar_10_specs as specs
    else:
        parser.error("Unknown dataset!")
        exit()

    # AE loss setting from arguments
    if args.loss == "f1":
        from losses import F1Loss
        autoencoder_loss = F1Loss()
    elif args.loss == "jaccard":
        from losses import JaccardLoss
        autoencoder_loss = JaccardLoss()
    elif args.loss == "BCE":
        from losses import BCEWithLogitsLoss
        autoencoder_loss = BCEWithLogitsLoss()
    elif args.loss == "CE":
        from losses import CrossEntropyLoss
        autoencoder_loss = CrossEntropyLoss()
    else:
        from losses import MSELoss
        autoencoder_loss = MSELoss()

    if args.decoder_input == "MIXED_TEACHER_FORCING":
        assert args.teacher_forcing_probability > -1, "Set a teacher forcing probability"

    # Autoencoder setup from arguments
    n_batches = int(ceil(specs.n_samples / args.batch_size)) * args.p_epochs
    autoencoder_setup = get_ae_setup(args.ae_objective, args.autoencoder, specs.img_height, specs.img_width,
                                     args.embedding_size, specs.n_channels, autoencoder_loss, args.n_layers,
                                     args.decoder_input, args.teacher_forcing_probability, n_batches)

    if args.supervised:
        learner = SupervisedLearner(args, specs)
    else:
        learner = UnsupervisedLearner(args, specs, autoencoder_setup)
    learner.run_repeated_learning()
