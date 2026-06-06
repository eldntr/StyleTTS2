import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        plbert_config = yaml.safe_load(f)

    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    checkpoint_path = os.path.join(log_dir, "step_1000000.t7")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"PLBERT checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state']

    new_state_dict = {}
    for name, value in state_dict.items():
        if name.startswith('module.'):
            name = name[7:]
        if name.startswith('encoder.'):
            name = name[8:]
        new_state_dict[name] = value

    new_state_dict.pop("embeddings.position_ids", None)
    bert.load_state_dict(new_state_dict, strict=False)

    return bert