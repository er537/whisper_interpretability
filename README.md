# Interpreting OpenAI's Whisper

This repo contains the code used to produce:
https://er537.github.io/blog/2023/09/05/whisper_interpretability.html

## NB) This repo is part of ongoing research and thus is not guaranteed to always be in a working state

To get started clone the repo and make a venv:
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt

The repo is structured as follows:
 - **whisper_interpretability/sparse coding** contains the code to train autoencoders on the internal representations of Whisper to extract the features it learns (using the techniques in https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)
 - **whisper_interpretability/audiolize** contains the code used to collect maximally activating dataset examples for neurons in Whisper
