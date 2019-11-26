import numpy as np

import unittest

from silent_frames_evaluation import eval_silent_frames


class TestEvalSilentFrames(unittest.TestCase):

    def test_pes_simple(self):
        true = np.ones((40,)) * 2
        true[10:20] = np.zeros((10,))
        true[25:35] = np.zeros((10,))
        predicted = np.ones((40,)) * (-3)

        window_size = 10
        hop_size = 5

        pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true, predicted, window_size,
                                                                                           hop_size,
                                                                                           eval_incomplete_last_frame=False,
                                                                                           eps_for_silent_target=False)
        pes_expected = np.array([10 * np.log10(10 * (-3)**2 + 10**(-12)), 10 * np.log10(10 * (-3)**2 + 10**(-12))])

        correct_pes = np.array_equal(pes, pes_expected)

        self.assertEqual(correct_pes, True, "PES array not as expected")

    def test_eps_simple(self):
        true = np.ones((43,)) * 2
        predicted = np.ones((43,)) * 4
        predicted[5:15] = np.zeros((10,))
        predicted[30:43] = np.zeros((13,))

        window_size = 10
        hop_size = 5

        pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true, predicted, window_size,
                                                                                           hop_size,
                                                                                           eval_incomplete_last_frame=False,
                                                                                           eps_for_silent_target=False)
        eps_expected = np.array(
            [10 * np.log10(10 * 2 ** 2 + 10 ** (-12)), 10 * np.log10(10 * 2 ** 2 + 10 ** (-12))])

        correct_pes = np.array_equal(eps, eps_expected)

        self.assertEqual(correct_pes, True, "EPS array not as expected")

    def test_eps_for_silent_target(self):
        true = np.ones((100,)) * 2
        true[50:60] = np.zeros((10,))
        predicted = np.ones((100,)) * 2
        predicted[50:60] = np.zeros((10,))

        window_size = 10
        hop_size = 5

        eps_expected = -120

        pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true, predicted, window_size,
                                                                                           hop_size, False, True)

        self.assertEqual(eps, eps_expected, "EPS does not take silent target into account but it should")

    def test_eps_silent_target_last_frame(self):
        true = np.ones((43,)) * 2
        true[30:43] = np.zeros((13,))
        predicted = np.ones((43,)) * 4
        predicted[5:15] = np.zeros((10,))
        predicted[30:43] = np.zeros((13,))

        window_size = 10
        hop_size = 5

        pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true, predicted, window_size,
                                                                                           hop_size, True, True)

        eps_expected = np.array([10 * np.log10(10 * 2**2 + 10**(-12)), -120, -120])

        correct_eps = np.array_equal(eps, eps_expected)

        self.assertEqual(correct_eps, True, "EPS does not take the last frame into account but it should")

    def test_eps_no_silent_target_but_last_frame(self):
        true = np.ones((43,)) * 2
        true[30:40] = np.zeros((10,))
        predicted = np.ones((43,)) * 4
        predicted[5:15] = np.zeros((10,))
        predicted[30:43] = np.zeros((13,))

        window_size = 10
        hop_size = 5

        pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true, predicted, window_size,
                                                                                           hop_size,
                                                                                           eval_incomplete_last_frame=True,
                                                                                           eps_for_silent_target=False)

        eps_expected = np.array([10 * np.log10(10 * 2**2 + 10**(-12)), 10 * np.log10(3 * 2**2 + 10**(-12))])

        correct_eps = np.array_equal(eps, eps_expected)

        self.assertEqual(correct_eps, True, "EPS not as expected")


    def test_with_last_frame(self):
        true = np.ones((103,)) * 2
        true[10:20] = np.zeros((10,))
        true[45:55] = np.zeros((10,))
        predicted = np.ones((103,)) * 2
        predicted[50:60] = np.zeros((10,))
        predicted[70:80] = np.zeros((10,))
        predicted[95:103] = np.zeros(8, )
        window_size = 10
        hop_size = 5

        pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true, predicted, window_size,
                                                                                           hop_size, True, False)

        pes_expected = np.array([10 * np.log10(10 * 4 + 10 ** (-12)), 10 * np.log10(5 * 4 + 10 ** (-12))])
        eps_expected = np.array([10 * np.log10(5 * 4 + 10 ** (-12)), 10 * np.log10(10 * 4 + 10 ** (-12)),
                                 10 * np.log10(8 * 4 + 10 ** (-12))])

        correct_pes = np.array_equal(pes, pes_expected)
        correct_eps = np.array_equal(eps, eps_expected)

        silent_true_source_frames_expected = [2, 9]
        silent_prediction_frames_expected = [10, 14, 19]

        with self.subTest():
            self.assertEqual(correct_pes, True, "PES is not as expected")
        with self.subTest():
            self.assertEqual(correct_eps, True, "EPS is not as expected")
        with self.subTest():
            self.assertEqual(silent_true_source_frames, silent_true_source_frames_expected, "silent true source frames "
                                                                                            "not correctly detected")
        with self.subTest():
            self.assertEqual(silent_prediction_frames, silent_prediction_frames_expected, "Silent prediction frames not"
                                                                                          " correctly detected")

    def test_without_last_frame(self):
        true = np.ones((103,)) * 2
        true[10:20] = np.zeros((10,))
        true[45:55] = np.zeros((10,))
        predicted = np.ones((103,)) * (-2)
        predicted[50:60] = np.zeros((10,))
        predicted[70:80] = np.zeros((10,))
        predicted[95:103] = np.zeros(8, )
        window_size = 10
        hop_size = 5

        pes, eps, silent_true_source_frames, silent_prediction_frames = eval_silent_frames(true, predicted, window_size,
                                                                                           hop_size, False, False)
        pes_expected = 10 * np.log10(10 * 4 + 10 ** (-12)) + 10 * np.log10(5 * 4 + 10 ** (-12))
        eps_expected = 10 * np.log10(10 * 4 + 10 ** (-12)) + 10 * np.log10(5 * 4 + 10 ** (-12))
        silent_true_source_frames_expected = [2, 9]
        silent_prediction_frames_expected = [10, 14]

        pes_expected = np.array([10 * np.log10(10 * 4 + 10 ** (-12)), 10 * np.log10(5 * 4 + 10 ** (-12))])
        eps_expected = np.array([10 * np.log10(5 * 4 + 10 ** (-12)), 10 * np.log10(10 * 4 + 10 ** (-12))])

        correct_pes = np.array_equal(pes, pes_expected)
        correct_eps = np.array_equal(eps, eps_expected)

        with self.subTest():
            self.assertEqual(correct_pes, True, "PES is not as expected")
        with self.subTest():
            self.assertEqual(correct_eps, True, "EPS is not as expected")
        with self.subTest():
            self.assertEqual(silent_true_source_frames, silent_true_source_frames_expected, "silent true source frames "
                                                                                            "not correctly detected")
        with self.subTest():
            self.assertEqual(silent_prediction_frames, silent_prediction_frames_expected, "Silent prediction frames not"
                                                                                          " correctly detected")


if __name__ == '__main__':
    unittest.main()
