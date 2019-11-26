# Weakly Informed Audio Source Separation
Here we make available a PyTorch implementation of the audio source separation model and the evaluation metrics proposed in the paper "Weakly Informed Audio Source Separation" by Kilian Schulze-Forster, Clement Doire, Gaël Richard, Roland Badeau. Published in *IEEE Workshop on Applications of Signal Processing to Audio and Acoustics, 2019.*

The paper, audio examples, and further information are available [here](https://schufo.github.io/publication/2019-WASPAA)

## Requirements
<pre>
numpy==1.15.4
torch==1.0.1.post2
</pre>

## Source Separation Evaluation Metrics

In the paper, we propose the metrics **Predicted Energy at Silence (PES)** and **Energy at Predicted Silence (EPS)**. They measure separation quality on evaluation frames where the target source is silent or the prediction is silent. The standard BSS\_eval metrics SDR, SAR, SIR are not defined on those frames. This becomes an issue when the metrics are computed on 1 second long non-overlapping windows, which became the standard way for musical source separation. This is also the [evaluation procedure of SiSEC](https://arxiv.org/abs/1804.06267). This means for the evaluation of singing voice separation on the MUSDB18 test set that 45 out of 210 minutes are systematically ignored because they do not contain vocals. Therefore, we propose the PES and EPS as metrics to complement the BSS\_eval metrics.

**Predicted Energy at Silence (PES)** measures the energy (in dB) in the prediction for frames where the target source is silent. It indicates whether an algorithm confuses other source with the target when the target is silent. Lower values are better since PES measures energy in the prediction on parts which should have zero energy.

**Energy at Predicted Silence (EPS)** measures the energy (in dB) in the target source for frames where the algorithm predicts silence. It indicates whether an algorithm predicts silence at the correct time. Lower values are better since EPS measures the true energy for frames where zero energy was predicted.

### Implementation and Usage of Evaluation Metrics

You can clone the repository or simply copy the file `silent_frames_evaluation.py` in your project to use the evaluation metrics. Import the function `eval_silent_frames` into your evaluation script. You can do the evaluation as follows:

<pre>
pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true_source,
                                                                                   predicted_source,
										   window_size,
                                                                                   hop_size,
                                                                                   eval_incomplete_last_frame=False,
                                                                                   eps_for_silent_target=True)
</pre>

**INPUTS**

`true_source` should be a one-dimensional numpy array containing the true source signal in the time domain.

`predicted_source` should be a one-dimensional numpy array containing the prediction in the time domain. It must have the same length as `true_source`.

`window_size` is the length of the evaluation frames in samples. `hop_size` is the hop size of the evaluation window in samples. You should use the same values as you use for the standard metrics, for example in the [mir_eval implementation](https://craffel.github.io/mir_eval/#mir_eval.separation.bss_eval_sources_framewise). We usually use the number of samples corresponding to one second as window size and hop size (no overlap).

`eval_incomplete_last_frame` is a boolean indicating whether you want to evaluate the last frame if it is shorter than the window size. mir\_eval ignores the last frame, so we usually set `eval_incomplete_last_frame=False`.

`eps_for_silent_target` is a boolean indicating whether you want to compute the EPS for frames where also the target is silent. It makes totally sense to do this, so we usually set it to `True`, however, we did not do so in the evaluation in the experiments of the paper. Hence, we include this option for reproducibility.

**OUTPUTS**

`pes` is a numpy array containing the PES values for all applicable evaluation frames. We recommend to take the mean as representative value on the whole signal.

`eps` is a numpy array containing the EPS values for all applicable evaluation frames. We recommend to take the mean as representative value on the whole signal.

`silent_true_source_frames` is a list of indices (counting from 0) of evaluation frames with a silent target source. The PES is calculated on these frames.

`silent_prediction_frames` is a list of indices (counting from 0) of evaluation frames with silent a prediction. The EPS is calculated on these frames.


*For numerical reasons, we add $10^{-12}$ to the energy of all frames. Consequently, -120 dB is returned for frames with zero energy.*


## Source Separation Model

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

## Citing this work
If you use the evaluation metrics or the model in your work, please cite the paper:
<pre>
@inproceedings{schulze2019weakly,
  title={Weakly informed audio source separation},
  author={Schulze-Forster, Kilian and Doire, Cl{\'e}ment and Richard, Ga{\"e}l and Badeau, Roland},
  booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics},
  pages={268--272},
  year={2019}
}
</pre>

## Acknowledgment
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowsa-Curie grant agreement No. 765068.

## Copyright notice
Copyright 2019 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris.
All rights reserved.
