import math
import os
import pickle
from typing import List

import fire
import numpy as np
import torch
import torch.nn.functional as F
import whisper
from global_utils import device
from global_whisper_utils import load_audio, trim_audio
from jaxtyping import Float
from sparse_coding.local_utils import get_features
from whisper_interpretability.sparse_coding.test.max_activating_segments import MaxActivatingFragments
from torch import Tensor

MIN_MEL_AMP = -1.0
MAX_MEL_AMP = 2.0


def get_audio_paths(max_activating_fragments_path: str, num_examples: int = 10):
    with open(max_activating_fragments_path, "rb") as f:
        max_actv = pickle.load(f)
    audio_paths = []
    max_seq_pos = []
    for meta in max_actv.metadata.values():
        audio_paths.append(meta[0])
        max_seq_pos.append(torch.argmax(meta[3]).item())
    return audio_paths[:num_examples], torch.tensor(max_seq_pos[:num_examples])


def main(
    n_iters: int = 100,
    layer_id_to_use: str = "encoder.blocks.3",
    lr: float = 1e-1,
    out_dir: str = "/exp/ellenar/audiolize/optimized_inputs",
    save_every: int = 1000,
    feature_idx: int = None,
    num_features: int = 10,
    chk_path: str = None,
    deepdream: bool = True,
    max_activating_fragments_path: str = None,
):
    features, _ = get_features(feature_type="learnt", chk_path=chk_path, feature_idx=feature_idx)
    if num_features > 1:
        features = features[:, :num_features]  # [d_model, num_features]

    inputs = []
    if not deepdream:
        audio_paths, max_seq_pos = get_audio_paths(
            max_activating_fragments_path=max_activating_fragments_path
        )
        for audio_path in audio_paths:
            audio = load_audio(audio_path)
            # audio = trim_audio(audio.squeeze(), 980.08, 986.63)
            audio = whisper.pad_or_trim(audio.flatten())
            inputs.append(whisper.log_mel_spectrogram(audio))
        inputs = torch.stack(inputs, dim=0)
        inputs = (
            inputs.detach().to(device).requires_grad_(True)
        )  # make sure inputs.is_leaf==True and requires_grad==True. DO NOT change the order of detach and requires_grad
    else:
        # use empty input for deepdream
        inputs = torch.zeros(
            features.shape[1], 80, 3000, device=device, requires_grad=True
        )  # n_features, n_mels(must be 80), seq_len(must be 3000)
        max_seq_pos = None
    assert inputs.is_leaf
    whisper_model = whisper.load_model("tiny").to(device)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path_ext = f"lr_{lr}"
    if feature_idx is not None:
        out_path_ext += f"_feature_idx_{feature_idx}"
    if max_seq_pos is not None:
        torch.save(max_seq_pos, f"{out_dir}/{out_path_ext}_max_seq_pos.pt")

    optimize_input(
        inputs,
        whisper_model,
        layer_id_to_use,
        lr=lr,
        n_iters=n_iters,
        save_every=save_every,
        out_path=f"{out_dir}/{out_path_ext}",
        features=features,
        deepdream=deepdream,
        max_seq_pos=max_seq_pos,
    )


def optimize_input(
    inputs: List[Float[Tensor, "n_mels seq_len"]],
    model: torch.nn.Module,
    layer_id_to_use: str,
    lr: float,
    n_iters: int,
    save_every: int,
    out_path: str,
    features: Float[Tensor, "d_model"],
    deepdream: bool = False,
    max_seq_pos: Float[Tensor, "n_samples"] = None,
):
    global actvs_dict
    actvs_dict = {}
    for name, module in model.named_modules():
        if name == layer_id_to_use:
            hook_fn = get_activation_hook(name)
            hook = module.register_forward_hook(hook_fn)

    # Gradient ascent iterations
    min_activation = torch.min(inputs)
    max_activation = torch.max(inputs)
    for step in range(n_iters + 1):
        inputs = gradient_step(
            inputs,
            model,
            layer_id_to_use,
            lr=lr,
            min_activation=min_activation,
            max_activation=max_activation,
            max_seq_pos=max_seq_pos,
            deepdream=deepdream,
            ascent=True,
            features=features,
        )
        if step % save_every == 0:
            torch.save(inputs.detach().cpu(), f"{out_path}_step{step}.pt")
            print(f"Saved optimized inputs for layer {layer_id_to_use} to {out_path}_step{step}.pt")

    # Unregister hook
    hook.remove()

    return inputs


def get_activation_hook(layer_name: str):
    def hook_fn(model, input, output):
        actvs_dict[layer_name] = output

    return hook_fn


def gradient_step(
    inputs,
    model,
    layer_id_to_use,
    lr,
    min_activation=0.0,
    max_activation=1.0,
    deepdream: bool = False,
    ascent: bool = False,
    features: Float[Tensor, "d_model"] = None,
    max_seq_pos: Float[Tensor, "bsz"] = None,
):
    assert inputs.requires_grad
    assert inputs.is_leaf
    # Apply Gaussian blur to the inputs
    # inputs = GaussianBlur1d(sigma=3.0)(inputs)
    # we pass all the inputs through the model and cache the intermediate activations for all the.
    # We only optimize one 'batch' of the input per feature
    model.embed_audio(inputs)

    layer_activation = actvs_dict[layer_id_to_use]
    assert layer_activation.requires_grad
    if deepdream:
        loss = get_deepdream_loss(layer_activation, features)
    else:
        loss = 10 * get_feature_loss(
            layer_activation, feature=features, max_seq_pos=max_seq_pos
        )  # loss is very small so we multiply by 10 to prevent underflow
    loss.backward()
    print(loss)

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    if ascent:
        inputs = inputs + (lr * inputs.grad.data)
    else:
        inputs = inputs - (lr * inputs.grad.data)

    # Step 5: Clamp the data
    if not deepdream:
        inputs.clamp_(min_activation, max_activation)

    # clone and detach so that inputs.is_leaf==True
    return inputs.detach().clone().requires_grad_(True)


def get_deepdream_loss(
    actvs: Float[Tensor, "n_feats seq_len d_model"], features: Float[Tensor, "d_model n_feats"]
):
    target = actvs.clone()
    for feature_idx in range(features.shape[1]):
        target[feature_idx, :100, :] = features[
            :, feature_idx
        ]  # 100 frames corresponds to 2s of audio
    return torch.nn.MSELoss(reduction="mean")(actvs, target)


def get_feature_loss(
    actvs: Float[Tensor, "n_samples seq_len d_model"],
    feature: Float[Tensor, "d_model"],
    max_seq_pos: Float[Tensor, "n_samples"],
):
    target = actvs.clone()
    # maximize the activations of this feature around its max activating position
    for batch in range(actvs.shape[0]):
        target[batch, max_seq_pos[batch] - 20 : max_seq_pos[batch] + 20, :] = feature
    return torch.nn.MSELoss(reduction="mean")(actvs, target)


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
