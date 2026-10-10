import numpy as np
from sklearn.ensemble import RandomForestClassifier

from signal_utils import (generate_observation, extract_features, generate_dataset)

def train_ml_model(random_state = 42, training_samples = 5000):
    # Training a Random Forest across multiple noise levels
    rng = np.random.default_rng(random_state)

    # Varying noise during training to improve robustness.
    noise_levels = rng.uniform(0.25, 2.0, training_samples)

    X = []
    y = []

    for noise_std in noise_levels:
        signal_present = bool(rng.integers(0, 2))

        observation = generate_observation(signal_present = signal_present,\
                                           noise_std = noise_std, rng = rng)

        X.append(extract_features(observation))
        y.append(int(signal_present))

    X = np.asarray(X)
    y = np.asarray(y)

    model = RandomForestClassifier(n_estimators = 150,\
                                   max_depth = 12,\
                                   min_samples_leaf = 3,\
                                   class_weight = "balanced",\
                                   random_state = random_state,\
                                   n_jobs = -1)

    model.fit(X, y)

    return model

def calibrate_energy_threshold(noise_std, false_alarm_target = 0.05,\
                               calibration_samples = 5000, random_state = 123):
    # Setting an energy threshold using independent noise-only data
    rng = np.random.default_rng(random_state)
    energies = []

    for _ in range(calibration_samples):
        observation = generate_observation(signal_present = False,\
                                           noise_std = noise_std, rng = rng)

        energies.append(np.mean(observation ** 2))

    return float(np.quantile(energies, 1 - false_alarm_target))

def evaluate_detectors(model, noise_std, test_samples = 2000, threshold = None, random_state = 456):
    # Evaluating both detectors on the same held-out observations
    rng = np.random.default_rng(random_state)

    X_test = []
    y_test = []
    energy_test = []
    example_present = None
    example_absent = None

    for _ in range(test_samples):
        signal_present = bool(rng.integers(0, 2))

        observation = generate_observation(signal_present = signal_present,\
                                           noise_std = noise_std, rng = rng)

        features = extract_features(observation)

        X_test.append(features)
        y_test.append(int(signal_present))
        energy_test.append(features[0])

        if signal_present and example_present is None:
            example_present = observation.copy()

        if not signal_present and example_absent is None:
            example_absent = observation.copy()

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    energy_test = np.asarray(energy_test)

    if threshold is None:
        threshold = calibrate_energy_threshold(noise_std)

    ml_predictions = model.predict(X_test).astype(int)
    energy_predictions = (energy_test > threshold).astype(int)

    return {
        "y_true": y_test,
        "ml_predictions": ml_predictions,
        "energy_predictions": energy_predictions,
        "threshold": threshold,
        "example_present": example_present,
        "example_absent": example_absent,
    }
