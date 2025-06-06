## 📊 MFCC Visualization Dataset (`uav_mfcc_visualizations/`)

This folder contains MFCC (Mel-frequency cepstral coefficient) visualizations generated from raw UAV (drone) acoustic recordings. It includes **32 drone models**, each represented by multiple MFCC plot images organized in subfolders.

### 🗂 Folder Structure

```
dataset/uav_mfcc_visualizations/
├── Autel_Evo_II/
│   ├── Autel_Evo_II_1_mfcc.png
│   ├── ...
├── DJI_FPV/
│   ├── DJI_FPV_1_mfcc.png
│   ├── ...
├── Hover_X1/
│   └── ...
...
```

Each `.png` file represents one MFCC plot generated from a single `.wav` recording.

---

### 🔧 MFCC Extraction Parameters

The MFCC features were extracted using `librosa` with the following configuration:

```python
librosa.feature.mfcc(
    y=y,
    sr=sample_rate,
    n_mfcc=20,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)
```

---

### 🖼️ Visualization Details

- Plotting library: `matplotlib` + `librosa.display.specshow()`
- `figsize=(10, 4)`, `dpi=100`
- Colormap: `magma`
- **No axis ticks**, **no titles**, **no colorbars**
- Output format: `.png`

---

### 🔍 Use Case

This visualization dataset supports:

- UAV acoustic classification and pattern analysis  
- Data exploration and visual comparison across drone types  
- Image-based ML training (e.g., CNN on MFCC inputs)

---

### 📎 Citation/Attribution

If using this dataset, please cite or reference:

```
M. Wang and A. Berg, "UAV MFCC Visualization Dataset," 2025.
```