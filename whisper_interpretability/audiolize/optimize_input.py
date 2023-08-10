from typing import List

import fire
import torch
import whisper
from global_utils import device
from jaxtyping import Float
from torch import Tensor

MIN_MEL_AMP = -1.0
MAX_MEL_AMP = 2.0


def main(
    n_iters: int = 10,
    layers_ids_to_use: List[str] = ["encoder.blocks.3"],
    lr: float = 1e-3,
):
    inputs = torch.zeros(
        len(layers_ids_to_use), 80, 3000, device=device
    )  # n_layer_ids, n_mels(must be 80), seq_len(must be 3000)
    inputs.requires_grad = True
    whisper_model = whisper.load_model("tiny").to(device)
    optimized_inputs = optimize_input(
        inputs, whisper_model, layers_ids_to_use, lr=lr, n_iters=n_iters
    )
    return optimized_inputs


def optimize_input(
    inputs: List[Float[Tensor, "n_mels seq_len"]],
    model: torch.nn.Module,
    layer_ids_to_use: List[str],
    lr: float,
    n_iters: int,
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
    for _ in range(n_iters):
        inputs = gradient_ascent(inputs, model, layer_ids_to_use, lr=lr)

    # Unregister hooks
    for hook in hooks:
        hook.remove()

    print(inputs)
    return inputs


def get_activation_hook(layer_name: str, layer_idx: int):
    def hook_fn(model, input, output):
        actvs_dict[layer_name] = output[
            layer_idx, ...
        ]  # we only optimize one 'batch' of the input for each layer

    return hook_fn


def gradient_ascent(inputs, model, layer_ids_to_use, lr):
    # we pass all the inputs through the model and cache the intermediate activations for all the
    # layers of interest. We only optimize each input for a single layer though.
    model.embed_audio(inputs)

    losses = []
    for layer_id in layer_ids_to_use:
        layer_activation = actvs_dict[layer_id]
        loss = torch.nn.MSELoss(reduction="mean")(
            layer_activation, torch.zeros_like(layer_activation)
        )
        losses.append(loss)

    losses = torch.mean(torch.stack(losses, dim=0))
    losses.backward()

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    inputs = inputs + (lr * inputs.grad.data)

    # Step 5: Clamp the data
    inputs.clamp_(MIN_MEL_AMP, MAX_MEL_AMP)

    # clone and detach so that inputs.is_leaf==True
    return inputs.detach().clone().requires_grad_(True)


if __name__ == "__main__":
    fire.Fire(main)
