from datetime import datetime
import math
from pathlib import Path

import torch


class Learner:

    def __init__(self, args, specs):
        # Parameter setting from arguments
        self.n_pretrain_epochs = args.p_epochs
        self.n_finetuning_epochs = args.f_epochs
        self.lambda_ = args.lambda_                                      # Value of hyperparam lambda balancing ae and kmeans losses
        self.batch_size = args.batch_size                                # Size of the mini-batches used in the stochastic optimizer
        self.seeded = args.seeded                                        # Specify if runs are seeded
        self.n_runs = args.number_runs
        self.is_writing_to_disc = args.write_files
        self.dataset_name = args.dataset
        # Parameter setting from dataset specs
        self.dataset_path = specs.dataset_path
        self.n_channels = specs.n_channels
        self.n_clusters = specs.n_clusters
        self.trainset, self.testset = specs.trainset, specs.testset
        self.img_height, self.img_width = specs.img_height, specs.img_width
        # Setting derived parameters
        self.test_size = self.batch_size
        self.n_batches = int(math.ceil(len(self.trainset) / self.batch_size))    # Number of mini-batches
        self.n_test_batches = int(math.ceil(len(self.testset) / self.test_size))
        # Definition of the randomly-drawn (0-10000) seeds to be used for each run
        self.seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]
        self.start_time = datetime.now()
        self.kmeans_model = None
        self.embedding_size = args.embedding_size
        self.lr = args.lr
        self.run = 0
        # Hardware specifications
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.cpu:
            self.device = "cpu"
        self.inverse_disease_mapping = specs.inverse_disease_mapping if (
                self.dataset_name == "PTB" or
                self.dataset_name == "PTBMAT"
        ) else None
        self._setup_logging()

    def _log(self, content):
        print(content)
        if self.is_writing_to_disc:
            with self.logfile.open("a+") as f:
                f.write(f"{content}\n")

    def _setup_logging(self):
        now = datetime.now()
        time_format = "%Y_%m_%dT%H_%M"
        experiment_id = f"{self.dataset_name}_e_{self.n_pretrain_epochs}_f_{self.n_finetuning_epochs}_bs" \
                        f"_{self.batch_size}_" \
                        f"{now.strftime(time_format)}"
        self.directory = Path.cwd() / "metrics" / experiment_id
        if self.is_writing_to_disc:
            self.directory.mkdir()
        self.logfile = self.directory / "log.txt"

        self._log(f"Hyperparameters: lambda={self.lambda_}, pretrain_epochs={self.n_pretrain_epochs}, "
                  f"finetune_epochs={self.n_finetuning_epochs}, batch_size={self.batch_size}, "
                  f"initial_lr={self.lr}, n_runs={self.n_runs}, embedding_size={self.embedding_size}, "
                  f"dataset_path={self.dataset_path}")
