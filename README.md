# ECG Normal vs PVC Classifier

A Python-based project for **automatic classification of ECG signals** into **Normal** and **Premature Ventricular Contraction (PVC)** beats using **digital signal processing and machine learning**.

This project focuses on *signal-level understanding* rather than end-to-end deep learning, combining classical DSP techniques with interpretable feature extraction and classification.

---

## Project Overview

Electrocardiogram (ECG) signals provide vital information about the electrical activity of the heart.  
One common arrhythmia is **Premature Ventricular Contraction (PVC)** — an early heartbeat originating in the ventricles.

This project implements a full ECG processing pipeline:

1. **Signal filtering (0.5–40 Hz band-pass)**
2. **R-peak detection**
3. **Heartbeat segmentation**
4. **Feature extraction using autocorrelation + DCT**
5. **Beat classification using k-Nearest Neighbors**
6. **Visualization & GUI-based comparison**

Both **manual filter design** and **built-in SciPy filtering** are supported for comparison and educational purposes.

---

## Signal Processing Pipeline

#### Raw ECG
↓
#### Band-pass Filtering (Butterworth)
↓
#### R-Peak Detection
↓
#### Beat Segmentation
↓
#### Autocorrelation
↓
#### Discrete Cosine Transform (DCT)
↓
#### k-NN Classification (Normal / PVC)


---

## Feature Extraction

Each heartbeat is represented by:
- Normalized autocorrelation sequence
- First **20 DCT coefficients** (compact and robust representation)

This approach captures:
- Beat morphology
- Periodicity
- Shape differences between Normal and PVC beats

---

## GUI Features

The Tkinter-based GUI allows you to:

- Compare **manual vs built-in Butterworth filters**
- Visualize:
  - Raw ECG
  - Filtered ECG
  - Autocorrelation
- Display **confusion matrix and accuracy**
- Toggle filtering method interactively

---

## Project Structure

###### ECG-Normal-PVC-classifier/
###### │
###### ├── data/
###### │ ├── PVC_Train.txt
###### │ ├── PVC_Test.txt
###### │ ├── Normal_Train.txt
###### │ └── Normal_Test.txt
###### │
###### ├── main.py
###### ├── requirements.txt
###### ├── project_statement.png
###### └── results_summary.txt


---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
Main libraries used:

NumPy

SciPy

scikit-learn

Matplotlib

Tkinter (built-in with Python)

---
## How to Run
**GUI mode (default)**

```python main.py```



This launches an interactive interface for visualization and evaluation.

---

**CLI mode (optional)**

Uncomment ```main()``` and comment out ```launch_gui()``` at the bottom of main.py to run training directly from the terminal.

---

## Output

Classification accuracy

Confusion matrix

Detailed classification report

Saved plots in ```ecg_plots/```

Summary written to ```results_summary.txt```