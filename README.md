# whisper_interpretability
### NB) This repo is part of ongoing research and thus is not guaranteed to always be in a working state

To get started clone the repo and make a venv:
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt

The repo is structured as follows:
 - **probes** contains the code to train linear probes on the internal representations of Whisper in order to learn macroscopic information about what each layer knows
 - **sparse coding** contains the code to train autoencoders on the internal representations of Whisper to extract the features it learns (using the techniques in https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)
 - **feature_directions** contains more micelanous code to inpect the internal activations of Whisper such as an implementaion of logit lens and the ability to 'steer' representations by adding an activation vector (similar to https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector)
