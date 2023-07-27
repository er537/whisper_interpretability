import torch
import whisper
import numpy as np
from collections import defaultdict
from rich.table import Table
from rich.console import Console

from whisper_repo import Whisper
from global_whisper_utils import (
    WhisperActivationCache,
    get_mels_from_dblx,
    MULTILANG_DBLX_DICT,
)
from global_utils import device
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

NUM_SAMPLES = 10
NUM_LAYERS = 4
NUM_HEADS = 6


def get_lang_probability(activations, lang_code: str, model: torch.nn.Module):
    for name, param in model.named_parameters():
        if name == "decoder.token_embedding.weight":
            token_unembed = torch.transpose(param, 0, 1)
    for name, mod in model.named_modules():
        if name == "decoder.ln":
            ln = mod
    lang_token = tokenizer.encode(
        f"<|{lang_code}|>", allowed_special={f"<|{lang_code}|>"}
    )
    activations = ln(activations)
    logits = activations @ token_unembed.half()
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    return torch.nn.functional.softmax(logits, dim=-1)[:, lang_token]


def get_head_lang_probs(
    activations, lang_code: str, model: torch.nn.Module, layer: int
):
    for name, param in model.named_parameters():
        if name == "decoder.token_embedding.weight":
            token_unembed = torch.transpose(param, 0, 1)
    for name, mod in model.named_modules():
        if name == "decoder.ln":
            ln = mod
    for name, param in model.named_parameters():
        if name == f"decoder.blocks.{layer}.cross_attn.out.weight":
            attn_out_weight = param
        elif name == f"decoder.blocks.{layer}.cross_attn.out.bias":
            attn_out_bias = param
    lang_token = tokenizer.encode(
        f"<|{lang_code}|>", allowed_special={f"<|{lang_code}|>"}
    )
    head_probs = []
    for head in range(NUM_HEADS):
        head_out = (
            activations[:, :64] @ attn_out_weight[head * 64 : (head + 1) * 64, :].half()
        ) + attn_out_bias.half()
        head_out = ln(head_out)
        logits = head_out @ token_unembed.half()
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(tokenizer.all_language_tokens)] = False
        logits[:, mask] = -np.inf
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, lang_token]
        head_probs.append(probs.item())

    return torch.tensor(head_probs)


def run_logit_lens_on_layers():
    whisper_model = whisper.load_model("tiny")
    ModelDims = whisper_model.dims
    hacked_model = Whisper(ModelDims)
    hacked_model.load_state_dict(whisper_model.state_dict())
    score_dict = defaultdict(list)  # layer_name, score
    actv_mod = WhisperActivationCache(
        model=hacked_model,
        activations_to_cache=[
            f"decoder.blocks.{layer}.cross_attn.wv_hook" for layer in range(NUM_LAYERS)
        ],
    )
    for dblx_path, lang_code in MULTILANG_DBLX_DICT.items():
        mels = get_mels_from_dblx(dblx_path, NUM_SAMPLES)
        actv_mod.forward(mels)
        for layer in range(NUM_LAYERS):
            activations = actv_mod.activations[
                f"decoder.blocks.{layer}.cross_attn.wv_hook"
            ].to(device)
            lang_probs = get_lang_probability(activations, lang_code, hacked_model)
            score_dict[layer].append(torch.mean(lang_probs).item())
        actv_mod.reset_state()

    layerwise_dict = {}
    for layer, scores in score_dict.items():
        layerwise_dict[layer] = sum(scores) / len(scores)

    return layerwise_dict


def run_logit_lens_on_heads():
    whisper_model = whisper.load_model("tiny")
    ModelDims = whisper_model.dims
    hacked_model = Whisper(ModelDims)
    hacked_model.load_state_dict(whisper_model.state_dict())
    score_dict = defaultdict(list)  # layer_name, score
    actv_mod = WhisperActivationCache(
        model=hacked_model,
        activations_to_cache=[
            f"decoder.blocks.{layer}.cross_attn.wv_hook" for layer in range(NUM_LAYERS)
        ],
    )
    for dblx_path, lang_code in MULTILANG_DBLX_DICT.items():
        mels = get_mels_from_dblx(dblx_path, NUM_SAMPLES)
        actv_mod.forward(mels)
        for layer in range(NUM_LAYERS):
            activations = actv_mod.activations[
                f"decoder.blocks.{layer}.cross_attn.wv_hook"
            ].to(device)
            head_probs = get_head_lang_probs(
                activations, lang_code, hacked_model, layer
            )
            score_dict[layer].append(head_probs)
        actv_mod.reset_state()

    table = Table(title="Headwise Logit Attribution")
    table.add_column("Layer")
    for head in range(NUM_HEADS):
        table.add_column(f"Head {head}")
    for layer in range(NUM_LAYERS):
        layer_scores = torch.mean(torch.stack(score_dict[layer], dim=0), dim=0)
        table.add_row(
            f"Layer {layer}",
            *[f"{layer_scores[head].item():.3}" for head in range(NUM_HEADS)],
        )

    console = Console()
    console.print(table)


if __name__ == "__main__":
    score_dict = run_logit_lens_on_layers()
    print(score_dict)
    # run_logit_lens_on_heads()
