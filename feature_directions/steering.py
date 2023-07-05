import torch
import whisper
import torch
from collections import defaultdict

from utils import device


class SteeringModule:
    def __init__(self, model_name: str = "tiny", activations_to_steer: list = []):
        self.model_name = model_name
        self.class_labels = ["fr", "de"]
        self.model = whisper.load_model(model_name).to(device)
        self.activations_to_steer = activations_to_steer
        self.steering_vectors = self.get_stearing_vectors()
        self.activation_dir = "/exp/ellenar/whisper_activations"
        self.hooks = []

    def get_steering_vectors(self):
        steering_vectors = defaultdict(list)
        for name in self.activations_to_steer:
            for label in self.class_labels:
                activations = torch.load(
                    f"{self.activation_dir}/{self.model_name}_{name}_{label}"
                )
                steering_vectors[name].append(torch.mean(activations, dim=0))

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.activations_to_steer:
                self.hooks.append(
                    module.register_forward_hook(
                        get_steering_hook(
                            self.steering_vectors[name][0],
                            self.steering_vectors[name][1],
                        )
                    )
                )

    def forward(self, x):
        self.register_hooks()
        options = whisper.DecodingOptions(
            without_timestamps=False, fp16=(device == "cuda")
        )
        output = self.model.decode(x, options)
        self.remove_hooks()
        return output

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def get_steering_hook(pos_mean, neg_mean):
    def hook(module, input, output):
        output_ = output.detach()
        return output_ + pos_mean.half().to(device) - neg_mean.half().to(device)

    return hook
