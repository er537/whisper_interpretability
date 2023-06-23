import torch
from subprocess import CalledProcessError, run
import numpy as np
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def trim_audio(
    array: np.array,
    start_time: float,
    end_time: float,
    sample_rate: int = 16_000,
):
    """
    Trim the audio file base array to n_samples, as expected by the encoder.
    """
    start_frame = int(sample_rate * start_time)
    end_frame = int(sample_rate * end_time)
    assert len(array) > end_frame - start_frame

    return array[start_frame:end_frame]


def load_audio(file: str, sample_rate_hz: int = 16_000):
    """
    Taken from Whisper repo: https://github.com/openai/whisper/blob/main/whisper/audio.py

    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sample_rate_hz: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate_hz),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
