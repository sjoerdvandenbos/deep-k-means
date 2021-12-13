import argparse
import os

from unsupervised_learning import UnsupervisedLearner
from supervised_learning import SupervisedLearner


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
    parser = argparse.ArgumentParser(description="Deep k-means algorithm")
    parser.add_argument("-d", "--dataset", type=str.upper,
                        help="Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", required=True)
    parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
    parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true')
    parser.add_argument("-l", "--lambda", type=float, default=0.1, dest="lambda_",
                        help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")
    parser.add_argument("-e", "--p-epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("-f", "--f-epochs", type=int, default=100, help="Number of fine-tuning epochs per alpha value")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="Size of the minibatches used by the optimizer")
    parser.add_argument("-n", "--number_runs", type=int, default=1,
                        help="number of repetitions of the entire experiment")
    parser.add_argument("-a", "--autoencoder", type=str.upper, default="CONVOAUTOENCODER",
                        choices=["FCAUTOENCODER", "OLMAUTOENCODER", "RESNETCONVOAUTOENCODER",
                                 "RESNETCONVOAUTOENCODER18", "RESNETCONVOAUTOENCODER34",
                                 "RESNETCONVOAUTOENCODER50", "CONVOAUTOENCODER", "RESNETCONVOAUTOENCODER10", "LSTM"],
                        nargs="?", help="type of autoencoder to use")
    parser.add_argument("-w", "--write_files", default=False, action="store_true",
                        help="if enabled, will write files to disk")
    parser.add_argument("-o", "--loss", type=str, default="MSE", help="type of loss function")
    parser.add_argument("--supervised", action="store_true")
    args = parser.parse_args()

    # Dataset setting from arguments
    if args.dataset == "MNIST":
        from datasets_specs import mnist_specs as specs
    elif args.dataset == "PTB":
        from datasets_specs import ptb_img_specs as specs
    elif args.dataset == "PTBMAT":
        from datasets_specs import ptb_matrix_specs as specs
    elif args.dataset == "MIT":
        from datasets_specs import mit_specs as specs
    elif args.dataset == "CIFAR10":
        from datasets_specs import cifar_10_specs as specs
    else:
        parser.error("Unknown dataset!")
        exit()

    # AE loss setting from arguments
    if args.loss == "f1":
        from losses import f1_loss
        autoencoder_loss = f1_loss
    elif args.loss == "jaccard":
        from losses import jaccard_loss
        autoencoder_loss = jaccard_loss
    elif args.loss == "BCE":
        from torch.nn import BCEWithLogitsLoss
        autoencoder_loss = BCEWithLogitsLoss()
    else:
        from losses import autoencoder_loss

    if args.supervised:
        learner = SupervisedLearner(args, specs)
    else:
        learner = UnsupervisedLearner(args, specs, autoencoder_loss)
    learner.run_repeated_learning()
