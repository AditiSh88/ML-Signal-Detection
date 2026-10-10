import numpy as np

# Signal configuration
FS = 1000                 # Sampling frequency in Hz
N_SAMPLES = 256           # Samples per observation
SIGNAL_AMPLITUDE = 1.0
SIGNAL_FREQUENCY = 125    # Signal frequency in Hz

# Sampling time vector
TIME = np.arange(N_SAMPLES) / FS

def generate_observation(signal_present, noise_std, rng):
    # Generating a noisy discrete-time observation
    noise = rng.normal(0, noise_std, N_SAMPLES)

    if signal_present:
        phase = rng.uniform(0, 2 * np.pi)

        signal = SIGNAL_AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQUENCY * TIME + phase)

    else:
        signal = np.zeros(N_SAMPLES)

    return signal + noise

def extract_features(observation):
    # Extracting time-domain and frequency-domain features
    energy = np.mean(observation ** 2)
    peak_amplitude = np.max(np.abs(observation))

    # Real FFT
    spectrum = np.fft.rfft(observation)
    power = np.abs(spectrum) ** 2

    # Converting FFT-bin indices into physical frequencies in Hz
    frequencies = np.fft.rfftfreq(N_SAMPLES, d = 1 / FS)

    # Frequency resolution is FS / N_SAMPLES
    # Selecting the target frequency and its two neighbouring bins
    target_bin = np.argmin(np.abs(frequencies - SIGNAL_FREQUENCY))

    start = max(0, target_bin - 2)
    end = min(len(power), target_bin + 3)

    target_power = np.sum(power[start:end])
    total_power = np.sum(power) + 1e-12

    spectral_concentration = target_power / total_power

    return [energy, peak_amplitude, spectral_concentration]

def generate_dataset(n_samples, noise_std, rng):
    # Generating labelled observations at a fixed noise level
    X = []
    y = []

    for _ in range(n_samples):
        signal_present = bool(rng.integers(0, 2))

        observation = generate_observation(signal_present = signal_present,\
                                           noise_std = noise_std, rng = rng)

        X.append(extract_features(observation))
        y.append(int(signal_present))

    return np.asarray(X), np.asarray(y)
