import whisper
import torch

from whisper_repo.model import Whisper
from global_whisper_utils import load_audio, trim_audio
from global_utils import device
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

WHISPER_MODEL: str = "tiny"


def get_backward_attn_hook(name, attn_grads):
    def hook_fn(module, input_, output_):
        attn_grads[name] = input_[0]

    return hook_fn


def get_forward_attn_hook(name, attn_scores):
    def hook_fn(module, input_, output_):
        attn_scores[name] = input_[0].detach()

    return hook_fn


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def get_encoder_attn_map(attn_scores, attn_grads):
    R = None
    for layer in attn_scores.keys():
        attn_score = attn_scores[layer]
        attn_grad = attn_grads[layer]
        cam = avg_heads(attn_score, attn_grad)
        if R is None:
            R = torch.eye(attn_score.shape[-1], attn_score.shape[-1])
        R += apply_self_attention_rules(R, cam)

    return R


def main():
    base_whisper_model = whisper.load_model(WHISPER_MODEL)
    ModelDims = base_whisper_model.dims
    model = Whisper(ModelDims)  # use our own model definition with attn hook points
    model.load_state_dict(base_whisper_model.state_dict())

    audio_path = "/data/artefacts/diarization/testsets/adobe/wavs/Hot_ones_snl.wav"
    audio = load_audio(audio_path)
    audio = trim_audio(audio, start_time=0.0, end_time=10.0)
    audio = whisper.pad_or_trim(audio.flatten())
    mels = torch.tensor(whisper.log_mel_spectrogram(audio)).to(device)

    attn_grads = {}
    for name, mod in model.named_modules():
        if "encoder" in name and "attn_hook" in name:
            mod.register_full_backward_hook(get_backward_attn_hook(name, attn_grads))

    attn_scores = {}
    for name, mod in model.named_modules():
        if "encoder" in name and "attn_hook" in name:
            mod.register_forward_hook(get_forward_attn_hook(name, attn_scores))

    # run forward and backwards pass on model
    model.to(device)
    mels = mels.to(device)
    output = model.embed_audio(mels.unsqueeze(0))
    output.backward(torch.ones_like(output))

    R = get_encoder_attn_map(attn_scores, attn_grads)

    return R
