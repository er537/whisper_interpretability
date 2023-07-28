import pytest
import torch

from global_whisper_utils import WhisperActivationCache


@pytest.fixture
def whisper_activation_cache():
    return WhisperActivationCache()


def test_activation_module_reset_state(whisper_activation_cache):
    whisper_activation_cache.activations = {"a": 1, "b": 2}
    whisper_activation_cache.reset_state()
    assert whisper_activation_cache.activations == {}


def test_activation_module_hooks(whisper_activation_cache):
    whisper_activation_cache.register_hooks()
    assert len(whisper_activation_cache.hooks) == 1
    whisper_activation_cache.remove_hooks()
    assert len(whisper_activation_cache.hooks) == 0


def test_activation_module_forward(whisper_activation_cache):
    mels = torch.empty(10, 80, 3000)
    whisper_activation_cache.forward(mels)
    assert len(whisper_activation_cache.activations) > 0
