"""
Network: MLP for frame-level phoneme recognition.
"""
import torch
import torch.nn as nn


def get_config():
    from scripts.config import get_config as _get
    return _get()


class Network(nn.Module):
    def __init__(self, input_size, output_size, config=None):
        super(Network, self).__init__()
        config = config or get_config()

        self.model = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

        if config.get("weight_initialization"):
            self.initialize_weights(config)

    def initialize_weights(self, config):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_name = config.get("weight_initialization", "kaiming_normal")
                if init_name == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif init_name == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif init_name == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif init_name == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif init_name == "uniform":
                    nn.init.uniform_(m.weight)
                else:
                    raise ValueError(f"Invalid weight_initialization: {init_name}")
                m.bias.data.fill_(0)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
