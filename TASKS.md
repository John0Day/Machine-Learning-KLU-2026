# Projektaufgaben – CNN Verkehrszeichen-Klassifizierung (GTSRB)

## Branch-Strategie

```
main                          ← nur stabile, fertige Versionen
└── dev                       ← Integration aller Features
    ├── task/01-project-setup
    ├── task/02-data-loading
    ├── task/03-data-preprocessing
    ├── task/04-baseline-model
    ├── task/05-model-improvement
    ├── task/06-evaluation
    └── task/07-report
```

**Workflow pro Task:**
1. Von `dev` einen neuen Branch erstellen: `git checkout -b task/XX-name`
2. Arbeiten, committen
3. Pull Request nach `dev` öffnen
4. Die andere Person reviewed → Merge

---

## Tasks

---

### Task 01 – Project Setup
**Branch:** `task/01-project-setup`
**Ziel:** Repository-Grundstruktur aufsetzen

**Aufgaben:**
- Ordnerstruktur anlegen:
  ```
  /data/          ← GTSRB Rohdaten (nicht ins Repo pushen!)
  /src/           ← Python-Skripte (model.py, dataset.py, train.py, evaluate.py)
  /notebooks/     ← Jupyter Notebooks für Experimente
  /models/        ← gespeicherte Modell-Gewichte (.pth)
  /results/       ← Plots, Metriken, Confusion Matrix
  ```
- `requirements.txt` mit allen Abhängigkeiten (torch, torchvision, numpy, matplotlib, seaborn, scikit-learn)
- `.gitignore` (data/, models/*.pth, __pycache__, .env)
- `README.md` mit Projektbeschreibung, Setup-Anleitung und Datensatz-Download-Link

**Definition of Done:** Repo ist geklont, `pip install -r requirements.txt` läuft fehlerfrei, Struktur steht.

---

### Task 02 – Data Loading
**Branch:** `task/02-data-loading`
**Ziel:** GTSRB-Datensatz laden und verstehen

**Aufgaben:**
- GTSRB über `torchvision.datasets.GTSRB` laden
- Klassenverteilung visualisieren (Balkendiagramm: Anzahl Bilder pro Klasse)
- Beispielbilder aus verschiedenen Klassen anzeigen
- Klassen-Mapping erstellen (Index → Verkehrszeichen-Name)
- Datensatz-Statistiken ausgeben (Anzahl Bilder, Bildgrößen, Auflösungsverteilung)

**Definition of Done:** Skript `src/dataset.py` läuft, Visualisierungen werden in `/results/` gespeichert.

---

### Task 03 – Data Preprocessing
**Branch:** `task/03-data-preprocessing`
**Ziel:** Bilder für das CNN vorbereiten

**Aufgaben:**
- Alle Bilder auf einheitliche Größe bringen (32×32 oder 64×64 px)
- Normalisierung: Pixelwerte 0–255 → 0.0–1.0 (mit Mittelwert und Standardabweichung des Trainingssets)
- Data Augmentation für Trainingsdaten:
  - Zufällige Rotation (±15°)
  - Zufällige Helligkeits-/Kontrastveränderung
  - Zufälliges horizontales Spiegeln (nur bei symmetrischen Zeichen sinnvoll!)
- Train / Validation / Test Split (z.B. 70% / 15% / 15%)
- PyTorch `DataLoader` mit Batches (Batch-Size z.B. 64)

**Definition of Done:** `DataLoader` für Train/Val/Test läuft, Batch-Shape wird korrekt ausgegeben.

---

### Task 04 – Baseline Model
**Branch:** `task/04-baseline-model`
**Ziel:** Erstes funktionierendes CNN implementieren und trainieren

**Architektur (Baseline):**
```
Input (3 × 32 × 32)
→ Conv(32 Filter, 3×3) + ReLU + MaxPool(2×2)
→ Conv(64 Filter, 3×3) + ReLU + MaxPool(2×2)
→ Flatten
→ Linear(256) + ReLU + Dropout(0.5)
→ Linear(43)  ← 43 Klassen
→ Softmax
```

**Aufgaben:**
- Modell in `src/model.py` definieren (PyTorch `nn.Module`)
- Trainingsloop in `src/train.py` (Adam-Optimizer, Cross-Entropy-Loss)
- Validation-Loss und Accuracy nach jeder Epoche ausgeben
- Loss-Kurve plotten (Train vs. Validation)
- Modell-Gewichte speichern (`models/baseline.pth`)

**Definition of Done:** Modell trainiert ohne Fehler, Validation Accuracy > 80% erreichbar.

---

### Task 05 – Model Improvement
**Branch:** `task/05-model-improvement`
**Ziel:** Baseline verbessern und verschiedene Varianten vergleichen

**Aufgaben:**
- Variante A: Tieferes Netz (4–5 Conv-Layer)
- Variante B: Batch Normalization nach jeder Conv-Schicht
- Variante C: Transfer Learning mit vortrainiertem MobileNet (`torchvision.models`)
- Learning Rate Scheduling (z.B. `StepLR` oder `ReduceLROnPlateau`)
- Vergleichstabelle: Accuracy / Trainingszeit / Modellgröße
- Bestes Modell speichern

**Definition of Done:** Mindestens 3 Varianten verglichen, Ergebnisse in `results/model_comparison.csv`.

---

### Task 06 – Evaluation
**Branch:** `task/06-evaluation`
**Ziel:** Modell gründlich auswerten und Schwachstellen analysieren

**Aufgaben:**
- Test Set Accuracy des besten Modells
- Confusion Matrix (Heatmap) → welche Klassen werden verwechselt?
- Precision, Recall, F1-Score pro Klasse (`sklearn.metrics.classification_report`)
- **Bias-Analyse:** Vergleich Accuracy für häufige vs. seltene Klassen
- Beispiele für falsch klassifizierte Bilder visualisieren
- Grad-CAM Visualisierung: welche Bildbereiche aktiviert das Modell?
- Robustheitstest: Modell auf verrauschten / unscharfen Bildern testen

**Definition of Done:** Alle Plots und Metriken in `/results/`, Ergebnisse sind reproduzierbar.

---

### Task 07 – Report
**Branch:** `task/07-report`
**Ziel:** Schriftlichen Bericht verfassen (3000–5000 Wörter)

**Aufgaben:**
- Bericht als `report.md` oder `report.pdf` im Repo-Root
- Inhalt gemäß Vorgabe:
  1. **Einleitung:** Problembeschreibung und Motivation
  2. **Datensatz:** GTSRB Beschreibung, Klassenverteilung, Biases im Datensatz
  3. **Methode:** CNN-Architektur, Preprocessing, Training (keine Code-Details, Konzepte erklären)
  4. **Experimente:** Vergleich der Modellvarianten, Hyperparameter
  5. **Ergebnisse:** Accuracy, Confusion Matrix, Bias-Analyse, Grad-CAM
  6. **Diskussion:** Stärken, Schwächen, Limitierungen, mögliche Verbesserungen
  7. **Fazit**

**Definition of Done:** Bericht ist im Repo, 3000–5000 Wörter, **vor der Präsentationssitzung** eingereicht.

---

## Aufgabenverteilung (2 Personen)

| Person A | Person B |
|----------|----------|
| Task 01 – Project Setup | Task 03 – Data Preprocessing |
| Task 02 – Data Loading | Task 05 – Model Improvement |
| Task 04 – Baseline Model | Task 07 – Report |
| Task 06 – Evaluation (gemeinsam) | Task 06 – Evaluation (gemeinsam) |

> Beide Personen sollten Commits, Issues und Pull Requests haben – das wird individuell bewertet!

---

## Nützliche Befehle

```bash
# Neuen Task-Branch erstellen
git checkout dev
git pull origin dev
git checkout -b task/02-data-loading

# Änderungen committen
git add src/dataset.py
git commit -m "feat: add GTSRB data loader with class distribution plot"

# Branch pushen und PR öffnen
git push origin task/02-data-loading
```
