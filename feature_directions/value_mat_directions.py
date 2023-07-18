import torch
import whisper

from probes.train.dataset import MultiClassDataset
from probes.utils.activation_caches import WhisperActivationCache
from util import device, get_mels_from_dblx

DBLX_DICT = {
    "/data/artefacts/am/de/v2023.02_q1rebuild/data_train/train.dblx": "de",
    "/data/artefacts/am/fr/new_normaliser/data_train/train.dblx": "fr",
}  # file_path, lang_code
NUM_SAMPLES = 100
BATCH_SIZE = 10


def get_top_feat_dims(layer_idx):
    model_size = "tiny"
    model = whisper.load_model(model_size)
    for name, param in model.named_parameters():
        if name == "decoder.blocks.1.cross_attn.value.weight":
            value_mat = param
    actv_cache = WhisperActivationCache(
        model=model, activations_to_cache=[f"encoder.blocks.{layer_idx}"]
    )
    score_dict = {}
    for dblx_path, lang in DBLX_DICT.items():
        lang_scores = []
        mels = get_mels_from_dblx(dblx_path, NUM_SAMPLES)
        for batch in range(mels.shape[0] // BATCH_SIZE):
            output = actv_cache.forward(
                mels[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE, :, :].to(device)
            )
            if output[0].language == lang:
                activs = actv_cache.activations[f"encoder.blocks.{layer_idx}"]
                actv_cache.reset_state()
                activs = activs.mean(dim=1)
                activs = activs / torch.norm(activs, dim=-1).unsqueeze(-1)
                feat_dim_scores = activs.to(device) @ value_mat.half()
                lang_scores.append(feat_dim_scores.cpu())
        score_dict[lang] = torch.cat(lang_scores, dim=0).mean(dim=0).squeeze()

    return score_dict


if __name__ == "__main__":
    feat_dim_scores = get_top_feat_dims(1)
    vals = list(feat_dim_scores.values())
    print(torch.topk(vals[0], k=10), torch.topk(vals[1], k=10))
