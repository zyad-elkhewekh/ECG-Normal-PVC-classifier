import os
import numpy as np
from scipy import signal
from scipy.fftpack import dct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FS = 360
LOWCUT = 0.5
HIGHCUT = 40.0
DCT_COEFFS = 20
PLOT_DIR = "ecg_plots"

FILES = {
    "PVC_train": "data/PVC_Train.txt",
    "PVC_test": "data/PVC_Test.txt",
    "NORM_train": "data/Normal_Train.txt",
    "NORM_test": "data/Normal_Test.txt"
}
############################################################################################
def manual_butterworth(x, fs=360.0, f1=0.5, f2=40.0, order=4):
    x = np.asarray(x, dtype=np.float64)
    T = 1.0 / fs

    omega1 = (2/T) * np.tan(np.pi * f1 / fs)
    omega2 = (2/T) * np.tan(np.pi * f2 / fs)

    B = omega2 - omega1
    omega0 = np.sqrt(omega1 * omega2)

    lp_poles = []
    for k in range(order):
        theta = np.pi/2 + (2*k + 1) * np.pi / (2*order)
        lp_poles.append(np.exp(1j * theta))
    lp_poles = np.array(lp_poles)

    bp_poles = []
    for p in lp_poles:
        root = np.sqrt((B*p)**2 - 4*(omega0**2))
        bp_poles.append((B*p + root) / 2)
        bp_poles.append((B*p - root) / 2)

    bp_poles = np.array(bp_poles)

    bp_zeros = np.concatenate([
        np.zeros(order),
        np.full(order, np.inf)
    ])

    z_poles = (2/T + bp_poles) / (2/T - bp_poles)

    z_zeros = []
    for z in bp_zeros:
        if np.isinf(z):
            z_zeros.append(-1.0)
        else:
            z_zeros.append((2/T + z) / (2/T - z))
    z_zeros = np.array(z_zeros)

    b = np.real(np.poly(z_zeros))
    a = np.real(np.poly(z_poles))

    w0 = 2*np.pi * np.sqrt(f1*f2) / fs
    ejw = np.exp(1j*w0*np.arange(len(b)))
    H0 = np.sum(b * ejw) / np.sum(a * ejw)
    b = b / np.abs(H0)

    y = np.zeros_like(x)
    for n in range(len(x)):
        for k in range(len(b)):
            if n - k >= 0:
                y[n] += b[k] * x[n - k]
        for k in range(1, len(a)):
            if n - k >= 0:
                y[n] -= a[k] * y[n - k]
        y[n] /= a[0]

    return y
#################################################################################

def load_txt_signal(path):
    """Load a text ECG file that contains numbers separated by |, \n or whitespace."""
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = raw.replace('\n', '|').replace(',', '|')
    parts = [p.strip() for p in raw.split('|') if p.strip() != ""]
    arr = np.array([float(x) for x in parts], dtype=np.float64)
    return arr

def bandpass_filter(x, fs=FS, low=LOWCUT, high=HIGHCUT, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)

def normalize(x):
    xm = x - np.mean(x)
    s = np.std(xm)
    if s < 1e-8:
        return xm
    return xm / s

def detect_r_peaks(ecg, fs=FS):
    """Lightweight R-peak detector."""
    diff = np.diff(ecg, prepend=ecg[0])
    squared = diff ** 2
    win = int(0.12 * fs)
    if win < 1:
        win = 1
    ma = np.convolve(squared, np.ones(win) / win, mode='same')
    thresh = np.median(ma) * 4.0 + 1e-8
    min_distance = int(0.25 * fs)
    peaks, _ = signal.find_peaks(ma, distance=min_distance, height=thresh)
    return peaks

def extract_beats(record, peaks, fs=FS, pre_ms=200, post_ms=400):
    pre = int(pre_ms * fs / 1000)
    post = int(post_ms * fs / 1000)
    beats = []
    for p in peaks:
        start = p - pre
        end = p + post
        if start >= 0 and end <= len(record):
            beats.append(record[start:end])
    return beats

def autocorr_features(beat, nlags=200):
    x = beat - np.mean(beat)
    ac = np.correlate(x, x, mode='full')
    mid = len(ac) // 2
    end = min(mid + nlags, len(ac))
    pos = ac[mid:end]
    if np.max(np.abs(pos)) > 0:
        pos = pos / np.max(np.abs(pos))
    c = dct(pos, norm='ortho')
    if len(c) < DCT_COEFFS:
        c = np.pad(c, (0, DCT_COEFFS - len(c)))
    return c[:DCT_COEFFS]

def save_example_plots(raw, filtered, beat, tag):
    os.makedirs(PLOT_DIR, exist_ok=True)

    safe_tag = os.path.basename(tag).replace(".", "_").replace("|", "_")

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    t_raw = np.arange(len(raw)) / FS
    t_f = np.arange(len(filtered)) / FS

    axs[0].plot(t_raw, raw)
    axs[0].set_title("Raw signal")

    axs[1].plot(t_f, filtered)
    axs[1].set_title("Filtered (0.5–40 Hz Butterworth)")

    axs[2].plot(beat)
    axs[2].set_title("Beat segment")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{safe_tag}_waveforms.png")
    plt.close(fig)

    ac = np.correlate(beat - np.mean(beat), beat - np.mean(beat), mode='full')
    mid = len(ac) // 2
    pos = ac[mid:mid + 200]
    d = dct(pos, norm='ortho')

    fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6))
    ax2[0].plot(pos)
    ax2[0].set_title("Autocorrelation")

    ax2[1].stem(np.arange(len(d))[:DCT_COEFFS], d[:DCT_COEFFS])
    ax2[1].set_title("DCT (first coefficients)")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{safe_tag}_ac_dct.png")
    plt.close(fig2)

def process_file(path, label, use_manual=False):
    raw = load_txt_signal(path)
    print(f"Loaded '{path}', {len(raw)} samples")

    if use_manual:
        filtered = manual_butterworth(raw)
    else:
        filtered = bandpass_filter(raw)


    features = []
    labels = []

    if len(filtered) < 5 * FS:
        beat = normalize(filtered)
        features.append(autocorr_features(beat))
        labels.append(label)
        save_example_plots(raw, filtered, beat, path)
        return np.array(features), np.array(labels)

    peaks = detect_r_peaks(filtered)
    print(f"Detected {len(peaks)} R-peaks in {os.path.basename(path)}")

    beats = extract_beats(filtered, peaks)
    if len(beats) == 0:
        w = int(0.6 * FS)
        beats = [filtered[i:i + w] for i in range(0, len(filtered) - w, w)]
        print(f"Fallback segmentation: {len(beats)} beats")

    for i, b in enumerate(beats):
        b_norm = normalize(b)
        features.append(autocorr_features(b_norm))
        labels.append(label)
        if i < 2:
            save_example_plots(raw, filtered, b, f"{path}_beat{i}")

    return np.array(features), np.array(labels)

def build_dataset(files, use_manual=False):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for f, lab in [(files["PVC_train"], 1), (files["NORM_train"], 0)]:
        feats, labs = process_file(f, lab, use_manual)
        X_train.append(feats)
        y_train.append(labs)

    for f, lab in [(files["PVC_test"], 1), (files["NORM_test"], 0)]:
        feats, labs = process_file(f, lab, use_manual)
        X_test.append(feats)
        y_test.append(labs)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=13)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=["Normal", "PVC"])

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print(rep)

    with open("results_summary.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write(rep)

    return clf

def main():
    for k, v in FILES.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"Expected file '{v}' not found.")

    choice = input("manual or built in? [m/b]: ").lower()
    use_manual = (choice == "m")
    
    X_train, y_train, X_test, y_test = build_dataset(FILES, use_manual)
    train_and_evaluate(X_train, y_train, X_test, y_test)
    print("Done! Plots saved in ecg_plots/ and results saved in results_summary.txt")


# ============================================================
def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    return fig

def evaluate_model(use_manual):
    X_train, y_train, X_test, y_test = build_dataset(FILES, use_manual)

    clf = KNeighborsClassifier(n_neighbors=13)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return cm, acc

def plot_ecg_comparison(signal, title):
    raw = signal
    built_in = bandpass_filter(raw)
    manual = manual_butterworth(raw)

    t = np.arange(len(raw)) / FS

    fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(t, raw)
    axs[0].set_title("Raw ECG")

    axs[1].plot(t, built_in)
    axs[1].set_title("Built-in Butterworth (0.5–40 Hz)")

    axs[2].plot(t, manual)
    axs[2].set_title("Manual Butterworth (0.5–40 Hz)")

    peaks = detect_r_peaks(built_in)
    beats = extract_beats(built_in, peaks)

    if len(beats) > 0:
        beat = normalize(beats[0])
        ac = np.correlate(beat - np.mean(beat), beat - np.mean(beat), mode="full")
        mid = len(ac) // 2
        axs[3].plot(ac[mid:mid+200])
        axs[3].set_title("Autocorrelation (first beat)")
    else:
        axs[3].text(0.5, 0.5, "No beats detected", ha="center")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def launch_gui():
    root = tk.Tk()
    root.title("ECG Filter Comparison (Manual vs Built-in)")
    root.geometry("1000x800")

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    canvas_frame = ttk.Frame(frame)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    def clear_canvas():
        for widget in canvas_frame.winfo_children():
            widget.destroy()

    use_manual = tk.BooleanVar(value=False)

    def show_confusion():
        clear_canvas()
        cm, acc = evaluate_model(use_manual.get())
        title = f"Confusion Matrix (Accuracy = {acc:.2f})"
        fig = plot_confusion_matrix(cm, ["Normal", "PVC"], title)

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_normal():
        clear_canvas()
        sig = load_txt_signal(FILES["NORM_train"])
        fig = plot_ecg_comparison(sig, "Normal Train Signal")
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_pvc():
        clear_canvas()
        sig = load_txt_signal(FILES["PVC_train"])
        fig = plot_ecg_comparison(sig, "PVC Train Signal")
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(pady=10)

    ttk.Button(btn_frame, text="Plot Normal Train", command=show_normal).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_frame, text="Plot PVC Train", command=show_pvc).pack(side=tk.LEFT, padx=10)

    ttk.Checkbutton(
        btn_frame,
        text="Use Manual Butterworth",
        variable=use_manual
    ).pack(side=tk.LEFT, padx=10)

    ttk.Button(
        btn_frame,
        text="Show Confusion Matrix",
        command=show_confusion
    ).pack(side=tk.LEFT, padx=10)

    ttk.Button(btn_frame, text="Quit", command=root.destroy).pack(side=tk.LEFT, padx=10)

    root.mainloop()
# ============================================================


if __name__ == "__main__":
    launch_gui()
    #main()