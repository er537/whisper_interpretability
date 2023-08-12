import math
import os
from typing import List

import fire
import numpy as np
import torch
import torch.nn.functional as F
import whisper
from global_utils import device
from global_whisper_utils import load_audio, trim_audio
from jaxtyping import Float
from torch import Tensor

MIN_MEL_AMP = -1.0
MAX_MEL_AMP = 2.0


def main(
    n_iters: int = 10,
    layers_ids_to_use: List[str] = ["encoder.blocks.3"],
    lr: float = 1e-1,
    out_dir: str = "/exp/ellenar/audiolize/optimized_inputs",
    audio_path: str = None,
    save_every: int = 5000,
):
    if audio_path is not None:
        audio = load_audio(audio_path)
        audio = trim_audio(audio.squeeze(), 980.08, 986.63)
        audio = whisper.pad_or_trim(audio.flatten())
        inputs = whisper.log_mel_spectrogram(audio)
        inputs = inputs.to(device).unsqueeze(0).repeat(len(layers_ids_to_use), 1, 1)
        inputs = inputs.detach().requires_grad_(
            True
        )  # make sure inputs.is_leaf==True and requires_grad==True. DO NOT chnage the order of these operations
    else:
        inputs = torch.zeros(
            len(layers_ids_to_use), 80, 3000, device=device, requires_grad=True
        )  # n_layer_ids, n_mels(must be 80), seq_len(must be 3000)
    assert inputs.is_leaf
    whisper_model = whisper.load_model("tiny").to(device)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path_ext = f"lr_{lr}"
    if audio_path is not None:
        out_path_ext += f'_{os.path.basename(audio_path).split(".")[-2]}'

    optimize_input(
        inputs,
        whisper_model,
        layers_ids_to_use,
        lr=lr,
        n_iters=n_iters,
        save_every=save_every,
        out_path=f"{out_dir}/{out_path_ext}",
    )


def optimize_input(
    inputs: List[Float[Tensor, "n_mels seq_len"]],
    model: torch.nn.Module,
    layer_ids_to_use: List[str],
    lr: float,
    n_iters: int,
    save_every: int,
    out_path: str,
):
    hooks = []
    global actvs_dict
    actvs_dict = {}
    for name, module in model.named_modules():
        if name in layer_ids_to_use:
            hook_fn = get_activation_hook(name, layer_ids_to_use.index(name))
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Gradient ascent iterations
    min_activation = torch.min(inputs)
    max_activation = torch.max(inputs)
    max_neuron_pos = None
    for step in range(n_iters):
        inputs, max_neuron_pos = gradient_ascent_step(
            inputs,
            model,
            layer_ids_to_use,
            lr=lr,
            min_activation=min_activation,
            max_activation=max_activation,
            max_neuron_pos=max_neuron_pos,
        )
        if step % save_every == 0:
            for i, layer_id in enumerate(layer_ids_to_use):
                torch.save(inputs[i, ...].detach().cpu(), f"{out_path}_{layer_id}_step{step}.pt")
                print(
                    f"Saved optimized inputs for layer {layer_id} to {out_path}_{layer_id}_step{step}.pt"
                )

    # Unregister hooks
    for hook in hooks:
        hook.remove()

    return inputs


def get_activation_hook(layer_name: str, layer_idx: int):
    def hook_fn(model, input, output):
        actvs_dict[layer_name] = output[
            layer_idx, ...
        ]  # we only optimize one 'batch' of the input for each layer

    return hook_fn


def gradient_ascent_step(
    inputs,
    model,
    layer_ids_to_use,
    lr,
    min_activation=0.0,
    max_activation=1.0,
    optim_type="neuron",
    max_neuron_pos=None,
    neuron_idx=1,
):
    assert inputs.requires_grad
    assert inputs.is_leaf
    # Apply Gaussian blur to the inputs
    # blurred_inputs = GaussianBlur1d(sigma=3.0)(inputs)
    # we pass all the inputs through the model and cache the intermediate activations for all the
    # layers of interest. We only optimize one 'batch' of the input per layer.
    model.embed_audio(inputs)

    losses = []
    for layer_id in layer_ids_to_use:
        layer_activation = actvs_dict[layer_id]
        if optim_type == "deepdream":
            loss = get_deepdream_loss(layer_activation)
        else:
            loss, max_neuron_pos = get_neuron_loss(
                layer_activation, neuron_idx=neuron_idx, max_neuron_pos=max_neuron_pos
            )
        loss = torch.nn.MSELoss(reduction="mean")(
            layer_activation, torch.zeros_like(layer_activation)
        )
        losses.append(loss)

    losses = torch.mean(torch.stack(losses, dim=0))
    losses.backward()

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    inputs = inputs + (lr * inputs.grad.data)

    # Step 5: Clamp the data
    inputs.clamp_(min_activation, max_activation)

    # clone and detach so that inputs.is_leaf==True
    return inputs.detach().clone().requires_grad_(True), max_neuron_pos


def get_deepdream_loss(actvs):
    return torch.nn.MSELoss(reduction="mean")(actvs, torch.zeros_like(actvs))


def get_neuron_loss(actvs, neuron_idx, max_neuron_pos):
    if max_neuron_pos is None:
        max_neuron_pos = torch.argmax(actvs[:, neuron_idx])
    target = actvs.clone()
    # maximize the activations of the neuron_idx around its max activating position
    target[max_neuron_pos - 5 : max_neuron_pos + 5, neuron_idx] = 0.0
    return torch.nn.MSELoss(reduction="mean")(actvs, target), max_neuron_pos


class GaussianBlur1d(torch.nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.0):
        super().__init__()
        kernel = self.get_1d_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel_1d", kernel, persistent=False)

    def forward(self, x: Float[Tensor, "bsz n_mels seq_len"]):
        padding = self.kernel_1d.shape[0] // 2  # Ensure that seq_len does not change
        out = F.conv1d(
            x,
            self.kernel_1d.repeat(x.shape[1], 1, 1).to(device),
            padding=padding,
            groups=x.shape[1],
        )
        return out

    @staticmethod
    def get_1d_gaussian_kernel(kernel_size: int, sigma: float, muu: float = 0.0):
        x = torch.linspace(-kernel_size, kernel_size, kernel_size)
        dst = torch.sqrt(x**2)

        # lower normal part of gaussian
        normal = 1 / (sigma * math.sqrt(2 * np.pi))

        # Calculating Gaussian filter
        gauss = torch.exp(-((dst - muu) ** 2 / (2.0 * sigma**2))) * normal
        return gauss / gauss.sum()


if __name__ == "__main__":
    fire.Fire(main)
