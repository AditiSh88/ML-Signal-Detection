import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

from signal_utils import (
    FS,
    N_SAMPLES,
    TIME,
    SIGNAL_FREQUENCY,
)
from model import (
    train_ml_model,
    calibrate_energy_threshold,
    evaluate_detectors,
)

st.set_page_config(
    page_title="ML-Based Signal Detection",
    layout="wide",
)

st.title("ML-Based Signal Detection")

st.write(
    "Detecting whether a known sinusoidal signal is " \
    "present in a noisy observation and comparing a " \
    "Machine Learning (Random Forest) detector with a" \
    " traditional energy-threshold detector"
)

st.sidebar.header("Detection Settings")

noise_std = st.sidebar.slider(
    "Noise standard deviation",
    min_value=0.25,
    max_value=2.0,
    value=0.75,
    step=0.25,
    help="Higher values represent stronger noise.",
)

test_samples = st.sidebar.slider(
    "Number of test observations",
    min_value=500,
    max_value=5000,
    value=2000,
    step=500,
)

target_pfa = st.sidebar.slider(
    "Target false-alarm rate",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    format="%.2f",
    help=(
        "The desired proportion of noise-only observations "
        "that the energy detector incorrectly identifies "
        "as containing a signal."
    ),
)

@st.cache_resource
def get_trained_model():
    return train_ml_model()

@st.cache_data
def get_energy_threshold(noise_level, false_alarm_target):
    return calibrate_energy_threshold(
        noise_std=noise_level,
        false_alarm_target=false_alarm_target,
        calibration_samples=5000,
        random_state=123,
    )

with st.spinner("Loading the Random Forest model..."):
    model = get_trained_model()

threshold = get_energy_threshold(noise_std, target_pfa)

st.subheader("Signal Configuration")

c1, c2, c3 = st.columns(3)

c1.metric("Sampling frequency", f"{FS} Hz")
c2.metric("Signal frequency", f"{SIGNAL_FREQUENCY} Hz")
c3.metric("Samples per observation", N_SAMPLES)

results = evaluate_detectors(
    model=model,
    noise_std=noise_std,
    test_samples=test_samples,
    threshold=threshold,
    random_state=456,
)

y_true = results["y_true"]
ml_predictions = results["ml_predictions"]
energy_predictions = results["energy_predictions"]

ml_accuracy = accuracy_score(y_true, ml_predictions)
energy_accuracy = accuracy_score(y_true, energy_predictions)

ml_precision = precision_score(
    y_true, ml_predictions, zero_division=0
)
energy_precision = precision_score(
    y_true, energy_predictions, zero_division=0
)

ml_recall = recall_score(
    y_true, ml_predictions, zero_division=0
)
energy_recall = recall_score(
    y_true, energy_predictions, zero_division=0
)

ml_cm = confusion_matrix(
    y_true, ml_predictions, labels=[0, 1]
)
energy_cm = confusion_matrix(
    y_true, energy_predictions, labels=[0, 1]
)

# Actual false-alarm and detection rates from the test observations.
noise_only = y_true == 0
signal_present = y_true == 1

ml_false_alarm = (
    np.mean(ml_predictions[noise_only] == 1)
    if np.any(noise_only) else 0.0
)
energy_false_alarm = (
    np.mean(energy_predictions[noise_only] == 1)
    if np.any(noise_only) else 0.0
)

ml_detection = (
    np.mean(ml_predictions[signal_present] == 1)
    if np.any(signal_present) else 0.0
)
energy_detection = (
    np.mean(energy_predictions[signal_present] == 1)
    if np.any(signal_present) else 0.0
)

ml_f1 = f1_score(y_true, ml_predictions, zero_division=0)
energy_f1 = f1_score(y_true, energy_predictions, zero_division=0)

ml_miss_rate = 1 - ml_detection
energy_miss_rate = 1 - energy_detection

st.header("Detector Performance")

st.write(
    f"Results for noise standard deviation **{noise_std:.2f}**, "
    f"using **{test_samples:,} test observations**."
)


st.write(
    "The energy threshold is chosen using observations that "
    "contain noise but no signal. We aim to keep false alarms "
    f"near **{target_pfa:.0%}**"
)

metric_data = {
    "Metric": [
        "Accuracy",
        "Precision",
        "Detection probability (Recall)",
        "False-alarm rate",
        "F1 Score",
        "Miss Rate",
    ],
    "Random Forest": [
        ml_accuracy,
        ml_precision,
        ml_detection,
        ml_false_alarm,
        ml_f1,
        ml_miss_rate,
    ],
    "Energy Detector": [
        energy_accuracy,
        energy_precision,
        energy_detection,
        energy_false_alarm,
        energy_f1,
        energy_miss_rate,
    ],
}

metric_df = pd.DataFrame(metric_data).set_index("Metric")

st.dataframe(
    metric_df.style.format("{:.2%}"),
    use_container_width=True,
)


st.caption(
    "Detection probability is the fraction of signal-containing "
    "observations correctly detected. The false-alarm rate is "
    "the fraction of noise-only observations incorrectly flagged "
    "as signals."
)

st.header("Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Random Forest")

    fig, ax = plt.subplots(figsize=(5, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=ml_cm,
        display_labels=["Signal Absent", "Signal Present"],
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    ax.set_title("Random Forest")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Energy Detector")

    fig, ax = plt.subplots(figsize=(5, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=energy_cm,
        display_labels=["Signal Absent", "Signal Present"],
    )
    disp.plot(
        ax=ax,
        cmap="Oranges",
        colorbar=False,
        values_format="d",
    )

    ax.set_title("Energy Detector")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

st.header("Example Noisy Observations")

st.write(
    "Comparing one observation containing a signal with one "
    "containing noise only. The time-domain plots show the "
    "sample amplitudes, while the frequency-domain plots show "
    "where the observation's energy lies across frequencies."
)

example_present = results["example_present"]
example_absent = results["example_absent"]

def plot_observation(observation, title):
    """Plotting an observation in time and frequency domains."""

    time_ms = TIME * 1000

    spectrum = np.abs(np.fft.rfft(observation)) / N_SAMPLES
    frequencies = np.fft.rfftfreq(N_SAMPLES, d=1 / FS)

    if len(spectrum) > 1:
        if N_SAMPLES % 2 == 0:
            spectrum[1:-1] *= 2
        else:
            spectrum[1:] *= 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    axes[0].plot(time_ms, observation, linewidth=1)
    axes[0].set_title(f"{title}: Time Domain")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frequencies, spectrum, linewidth=1)
    axes[1].set_title(f"{title}: Frequency Domain")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, FS / 2)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

col1, col2 = st.columns(2)

with col1:
    st.subheader("Signal Present")
    fig = plot_observation(
        example_present,
        "Signal Present",
    )
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Signal Absent")
    fig = plot_observation(
        example_absent,
        "Noise Only",
    )
    st.pyplot(fig)
    plt.close(fig)

st.header("What Did the ML Model Learn?")

st.write(
    "The Random Forest uses three features extracted from each "
    "observation to distinguish between signal-present and "
    "signal-absent cases."
)

feature_names = [
    "Mean-square energy",
    "Peak amplitude",
    "Spectral concentration",
]

importance = pd.Series(
    model.feature_importances_,
    index=feature_names,
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))

importance.plot(
    kind="barh",
    ax=ax,
)

ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Detection Probability vs Noise")

if st.button("Run noise-level comparison"):
    noise_levels = np.arange(0.25, 2.01, 0.25)
    sweep_rows = []

    progress = st.progress(0)

    for i, level in enumerate(noise_levels):
        level_threshold = get_energy_threshold(
            float(level),
            target_pfa,
        )

        level_results = evaluate_detectors(
            model=model,
            noise_std=float(level),
            test_samples=1500,
            threshold=level_threshold,
            random_state=900 + i,
        )

        actual = level_results["y_true"]

        for name, key in [
            ("Energy Detector", "energy_predictions"),
            ("Random Forest", "ml_predictions"),
        ]:
            pred = level_results[key]

            present_mask = actual == 1

            detection_probability = (
                np.mean(pred[present_mask] == 1)
                if np.any(present_mask)
                else 0.0
            )

            sweep_rows.append({
                "Noise standard deviation": level,
                "Detector": name,
                "Detection Probability": detection_probability,
            })

        progress.progress((i + 1) / len(noise_levels))

    progress.empty()

    sweep_df = pd.DataFrame(sweep_rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for name in ["Energy Detector", "Random Forest"]:
        subset = sweep_df[sweep_df["Detector"] == name]

        ax.plot(
            subset["Noise standard deviation"],
            subset["Detection Probability"],
            marker="o",
            label=name,
        )

    ax.set_xlabel("Noise Standard Deviation")
    ax.set_ylabel("Detection Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection Probability at Different Noise Levels")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        sweep_df.pivot(
            index="Noise standard deviation",
            columns="Detector",
            values="Detection Probability",
        ).round(3),
        use_container_width=True,
    )

st.subheader("Interpretation")

st.markdown(
    """
    - **Accuracy:** overall fraction of correct decisions.
    - **Detection probability:** fraction of actual signals detected.
    - **False alarm rate:** fraction of noise-only observations
      incorrectly labelled as signals.
    - **Miss rate:** fraction of actual signals missed.
    - **F1 score:** balance between precision and recall.
    """
)
