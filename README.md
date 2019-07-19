# Weakly Informed Audio Source Separation
This is a PyTorch implementation of the audio source separation model proposed in the paper "Weakly Informed Audio Source Separation" by Kilian Schulze-Forster, Clement Doire, Gaël Richard, Roland Badeau. Published in IEEE Workshop on Applications of Signal Processing to Audio and Acoustics, 2019. 

The paper, audio examples and further information are available [here](https://schufo.github.io/publication/2019-WASPAA)

### Requirements
<pre>
numpy==1.15.4
torch==1.0.1.post2
</pre>

To train the model, you can create an instance of the class InformedSeparatorWithAttention.

In the experiments for the paper, the model was used with the following parameters:
<pre>
separator = InformedSeparatorWithAttention(mix_features=513,
                                           mix_encoding_size=513,
                                           mix_encoder_layers=2,
                                           side_info_features=1,
                                           side_info_encoding_size=513,
                                           side_info_encoder_layers=2,
                                           connector_output_size=513,
                                           target_decoding_hidden_size=513,
                                           target_decoding_features=513,
                                           target_decoder_layers=2)
</pre>

### Acknowledgment
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowsa-Curie grant agreement No. 765068.

### Copyright notice
Copyright 2019 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris.
All rights reserved.
