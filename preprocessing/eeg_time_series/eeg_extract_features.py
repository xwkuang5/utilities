import numpy as np
import math

from .pyeeg import hfd, hjorth, spectral_entropy, svd_entropy, permutation_entropy
import scipy.stats


def extract_features(
        signal,
        fs=256,
        feature_dict={
            "PS": 30,
            "PR": [0.5, 4, 7, 13, 30],
            "hjorth_fractal_dimension": 3,
            "hjorth": None
        }):
    """Extract eeg features for a bulk of signals

    The default values are never used because they must be passed from the
    users.

    Parameters:
        signals         - (w, m) 2-D numpy array of float, where w corresponds
                          to the number of signals and m corresponds to the number
                          of samples in each signal
        fs              - int, sampling frequency
        feature_dict    - dictionary, mapping feature names to feature parameters

    Returns:
        features        - (N,) 1-D numpy array of float, where N corresponds to
                          the number of feautres in each signal
    """

    features = []

    signal_length = len(signal)
    signal_fft = np.fft.fft(signal)
    signal_amplitude_spectrum = np.abs(signal_fft)
    signal_power_spectrum = signal_amplitude_spectrum**2

    # maximum frequency of interest is hard coded to be 30
    signal_amplitude_spectrum_30Hz = signal_amplitude_spectrum[:math.floor(
        30 * signal_length / fs)]

    signal_first_order_diff = list(np.diff(signal))

    for feature_name in sorted(list(feature_dict.keys())):

        if feature_name == "PS":
            max_freq = feature_dict["PS"]
            feature = np.sum(signal_power_spectrum[0:math.floor(
                max_freq / fs * signal_length)])
            features.append(feature)

        elif feature_name == "PR":
            band = feature_dict["PR"]
            assert band != [], "Missing band parameter for calculation of power ratio"
            power = np.zeros(len(band) - 1)
            for freq_idx in range(len(band) - 1):
                freq = float(band[freq_idx])
                next_freq = float(band[freq_idx + 1])
                power[freq_idx] = np.sum(signal_power_spectrum[math.floor(
                    freq / fs * signal_length):math.floor(
                        next_freq / fs * signal_length)])

            power_ratio = power / np.sum(power)

            features += list(power_ratio)

        elif feature_name == "hjorth_fractal_dimension":
            hfd_param = feature_dict["hjorth_fractal_dimension"]
            features.append(hfd(signal, hfd_param))

        elif feature_name == "hjorth":
            hm, hc = hjorth(signal, signal_first_order_diff)
            features.append(hm)
            features.append(hc)

        elif feature_name == "spectral_entropy":
            band = feature_dict["spectral_entropy"]
            assert band != [], "Missing band parameter for calculation of power ratio"
            power = np.zeros(len(band) - 1)
            for freq_idx in range(len(band) - 1):
                freq = float(band[freq_idx])
                next_freq = float(band[freq_idx + 1])
                power[freq_idx] = np.sum(signal_power_spectrum[math.floor(
                    freq / fs * signal_length):math.floor(
                        next_freq / fs * signal_length)])

            power_ratio = power / np.sum(power)

            features.append(spectral_entropy(signal, band, fs, power_ratio))

        elif feature_name == "svd_entropy":
            embedding_lag, embedding_dimension = feature_dict["svd_entropy"]
            features.append(
                svd_entropy(signal, embedding_lag, embedding_dimension))

        elif feature_name == "permutation_entropy":
            permutation_order, embedding_lag = feature_dict[
                "permutation_entropy"]
            features.append(
                permutation_entropy(signal, permutation_order, embedding_lag))
        elif feature_name == "mean":
            """
            the following features are from On the classification of sleep states
            by means of statistical and spectral features from single channel
            Electroencephalogram (Hassan et al., 2015)
            """
            features.append(np.mean(signal))

        elif feature_name == "variance":
            features.append(np.var(signal))

        elif feature_name == "skewness":
            features.append(scipy.stats.skew(signal))

        elif feature_name == "kurtosis":
            features.append(scipy.stats.kurtosis(signal))

        elif feature_name == "spectral_flatness":
            features.append(
                scipy.stats.gmean(signal_amplitude_spectrum_30Hz) /
                np.mean(signal_amplitude_spectrum_30Hz))

        # TODO: should weights be divided by 30Hz?
        elif feature_name == "spectral_centroid":
            features.append(
                np.average(
                    signal_amplitude_spectrum_30Hz,
                    weights=np.arange(signal_amplitude_spectrum_30Hz.shape[0]))
                / np.sum(signal_amplitude_spectrum_30Hz))

        elif feature_name == "spectral_spread":
            spectral_centroid = np.average(
                signal_amplitude_spectrum_30Hz,
                weights=np.arange(signal_amplitude_spectrum_30Hz.shape[
                    0])) / np.sum(signal_amplitude_spectrum_30Hz)
            features.append(
                np.sum(
                    np.multiply(
                        np.square(
                            np.arange(signal_amplitude_spectrum_30Hz.shape[0])
                            - spectral_centroid),
                        signal_amplitude_spectrum_30Hz)) /
                np.sum(signal_amplitude_spectrum_30Hz))

        elif feature_name == "spectral_decrease":
            diff = np.diff(signal_amplitude_spectrum_30Hz)
            m = signal_amplitude_spectrum_30Hz.shape[0]
            features.append(
                np.sum(np.divide(diff, np.arange(1, m))) /
                np.sum(signal_amplitude_spectrum_30Hz))

        else:
            print("Unknown feature: {}".format(feature_name))

    return np.asarray(features)
