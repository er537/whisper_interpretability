from collections import defaultdict
from typing import List

import torch
import whisper
from global_utils import device

"""
Module to 'steer' intermediete activations in Whisper.
Works by adding (steering_factor*mean activation)
of the first class in class_labels and subtracting to second class.
Steering vectors must be precomputed and saved to disk - see save_mean_cls_activation
"""


class LanguageSteeringModule:
    def __init__(
        self,
        activations_to_steer: list,
        model_name: str = "tiny",
        model: torch.nn.Module = None,  # if you want to use a model other than the default
        steering_factor: float = 1.0,
        class_labels: list = [
            "fr",
            "de",
        ],  # first is the class you would like to steer it towards, second is the true class
    ):
        self.model_name = model_name
        self.class_labels = class_labels
        if model is None:
            self.model = whisper.load_model(model_name)
        else:
            self.model = model
        self.model = self.model.to(device)
        self.activations_to_steer = activations_to_steer
        self.activation_dir = "/exp/ellenar/whisper_activations"
        self.hooks = []
        self.steering_factor = steering_factor
        self.get_steering_vectors()

    def get_steering_vectors(self):
        self.steering_vectors = defaultdict(list)
        for name in self.activations_to_steer:
            for label in self.class_labels:
                activations = torch.load(f"{self.activation_dir}/{self.model_name}_{name}_{label}")
                self.steering_vectors[name].append(activations)

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.activations_to_steer:
                self.hooks.append(
                    module.register_forward_hook(
                        get_lang_steering_hook(
                            self.steering_vectors[name][0],
                            self.steering_vectors[name][1],
                            self.steering_factor,
                        )
                    )
                )

    def forward(self, x):
        self.register_hooks()
        options = whisper.DecodingOptions(without_timestamps=False, fp16=(device == "cuda"))
        output = self.model.decode(x, options)
        self.remove_hooks()
        return output

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def get_lang_steering_hook(steering_cls_vector, true_cls_vector, steering_factor):
    def hook(module, input, output):
        """
        Add the 'steering' class vector and subtract the 'true' class vector
        """
        print(steering_cls_vector.shape, true_cls_vector.shape, output.shape)
        output_ = output.detach()
        output_[:, :, :] = (
            output_[:, :, :]
            + (steering_factor * steering_cls_vector.half().to(device))
            - (steering_factor * true_cls_vector.half().to(device))
        )
        return output_

    return hook


class PhonemeSteeringModule:
    def __init__(
        self,
        activations_to_steer: List[str],
        phoneme_vectors: List[torch.Tensor],
        model_name: str = "tiny",
        model: torch.nn.Module = None,  # if you want to use a model other than the default
        steering_factor: float = 1.0,
    ):
        self.model_name = model_name
        self.phoneme_vectors = phoneme_vectors
        if model is None:
            self.model = whisper.load_model(model_name)
        else:
            self.model = model
        self.model = self.model.to(device)
        self.activations_to_steer = activations_to_steer
        self.activation_dir = "/exp/ellenar/whisper_activations"
        self.hooks = []
        self.steering_factor = steering_factor

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.activations_to_steer:
                self.hooks.append(module.register_forward_hook(self.get_phone_steering_hook()))

    def forward(self, x):
        self.register_hooks()
        options = whisper.DecodingOptions(
            without_timestamps=False, fp16=(device == "cuda"), language="en"
        )
        output = self.model.decode(x, options)
        self.remove_hooks()
        return output

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_phone_steering_hook(self):
        def hook(module, input, output):
            """
            Add the 25 frames of each phoneme vector to the output
            """
            for i, phoneme_vector in enumerate(self.phoneme_vectors):
                print(output.shape, "output shape")
                output[:, :, :] = phoneme_vector
            return output

        return hook
