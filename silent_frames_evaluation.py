import numpy as np


def eval_silent_frames(true_source, predicted_source, window_size: int, hop_size: int, eval_last_frame=False,
                       eps_for_silent_target=True):
    """
    :param true_source: true source signal in the time domain, numpy array with shape (T,)
    :param predicted_source: predicted source signal in the time domain, numpy array with shape (T,)
    :param window_size: length (in samples) of the window used for the framewise bss_eval metrics computation
    :param hop_size: hop size (in samples) used for the framewise bss_eval metrics computation
    :param eval_last_frame: if True, takes last frame into account even if it is shorter than the window, default: False
    :param eps_for_silent_target: if True, returns a value also if target source is silent, set to False for exact
    behaviour as explained in the paper "Weakly Informed Audio Source Separation", default: True
    :return: pes: numpy array containing PES values for all applicable frames
             eps: numpy array containing EPS values for all applicable frames
             silent_true_source_frames: list of indices of frames with silent target source
             silent_prediction_frames: list of indices of frames with silent predicted source
    """

    # check inputs
    assert true_source.ndim == 1, "true source array has too many dimensions, expected shape is (T,)"
    assert predicted_source.ndim == 1, "predicted source array has too many dimensions, expected shape is (T,)"
    assert len(true_source) == len(predicted_source), "true source and predicted source must have same length"

    # compute number of evaluation frames
    number_eval_frames = int(np.ceil((len(true_source) - window_size) / hop_size)) + 1

    last_frame_incomplete = False
    if len(true_source) % hop_size != 0:
        last_frame_incomplete = True

    # values for each frame will be gathered here
    pes_list = []
    eps_list = []

    # indices of frames with silence will be gathered here
    silent_true_source_frames = []
    silent_prediction_frames = []

    for n in range(number_eval_frames):

        # evaluate last frame if applicable
        if n == number_eval_frames - 1 and last_frame_incomplete:
            if eval_last_frame:
                prediction_window = predicted_source[n * hop_size:]
                true_window = true_source[n * hop_size:]
            else:
                continue

        # evaluate other frames
        else:
            prediction_window = predicted_source[n * hop_size: n * hop_size + window_size]
            true_window = true_source[n * hop_size: n * hop_size + window_size]

        # compute Predicted Energy at Silence (PES)
        if sum(abs(true_window)) == 0:
            pes = 10 * np.log10(sum(prediction_window ** 2) + 10 ** (-12))
            pes_list.append(pes)
            silent_true_source_frames.append(n)

        # compute Energy at Predicted Silence (EPS)
        if eps_for_silent_target:
            if sum(abs(prediction_window)) == 0:
                true_source_energy_at_silent_prediction = 10 * np.log10(sum(true_window ** 2) + 10 ** (-12))
                eps_list.append(true_source_energy_at_silent_prediction)
                silent_prediction_frames.append(n)

        else:
            if sum(abs(prediction_window)) == 0 and sum(abs(true_window)) != 0:
                true_source_energy_at_silent_prediction = 10 * np.log10(sum(true_window ** 2) + 10 ** (-12))
                eps_list.append(true_source_energy_at_silent_prediction)
                silent_prediction_frames.append(n)

    pes = np.asarray(pes_list)
    eps = np.asarray(eps_list)

    return pes, eps, silent_true_source_frames, silent_prediction_frames
