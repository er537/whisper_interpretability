
# Acoustic Features
We found max activating dataset examples for all of the neurons in the mlp layers of the encoder. We also trained sparse autoencoders on the residual stream activations and found max activating dataset examples for these learnt features. By amplifying the sound around the point where the neuron/feature is maximally active, we found them to be highly interpretable. Below are examples of some of the features we found.

## encoder.blocks.3.mlp.1 - Neuron Basis
<details>
<summary>Neuron 4 ('said')</summary>
<audio controls>
   <source src="audio/b3_mlp1_nb_4pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_4pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_4pkl_sample4.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_4pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 9 ('way/wait')</summary>
<audio controls>
   <source src="audio/b3_mlp1_nb_9pkl_sample5.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_9pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_9pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 20 ('f')</summary>
<audio controls>
   <source src="audio/b3_mlp1_nb_20pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_20pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_20pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_mlp1_nb_20pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

## encoder.blocks.2.mlp.1 - Neuron Basis
<details>
<summary>Neuron 0 ('m')</summary>
<audio controls>
   <source src="audio/b2_mlp1_nb_0pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_0pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_0pkl_sample7.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_0pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 1 ('sh/ch'))</summary>
<audio controls>
   <source src="audio/b2_mlp1_nb_1pkl_sample1.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_1pkl_sample2.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_1pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_1pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 3 ('c')</summary>
<audio controls>
   <source src="audio/b2_mlp1_nb_3pkl_sample4.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_3pkl_sample5.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_3pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b2_mlp1_nb_3pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

## encoder.blocks.3 - Learnt using sparse autoencoder
<details>
<summary>Dictionary idx=131 ("r")</summary>
<audio controls>
   <source src="audio/b3_res_131pkl_sample2.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_131pkl_sample7.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_131pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_131pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Dictionary idx=1 ("n")</summary>
<audio controls>
   <source src="audio/b3_res_1pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_1pkl_sample1.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_1pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_1pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Dictionary idx=2 ("p")</summary>
<audio controls>
   <source src="audio/b3_res_2pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_2pkl_sample2.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_2pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="audio/b3_res_2pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>


# The decoder is a weak LM

# The encoder representations are very localized
