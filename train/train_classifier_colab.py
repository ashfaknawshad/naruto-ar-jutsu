"""
Kekkai classifier training — Colab version.

Same architecture and evaluation methodology as train/train-classifier.mjs
(the local Node script), reimplemented with numpy + scikit-learn so it runs
in seconds instead of the ~15-25 minutes the hand-rolled pure-JS version
takes locally. Produces an identical weights.json — classifier.ts doesn't
need to know or care which one trained it.

Usage in Colab:
  1. New notebook, paste this whole file into one cell.
  2. Run it. A file-upload widget appears — select every file in your local
     datasets/raw/*.json (multi-select in the dialog).
  3. It prints the cross-validation report, then trains the final model and
     triggers a download of weights.json.
  4. Drop that file into public/models/weights.json in the repo, replacing
     the existing one.
"""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# --- Config — mirrors train/train-classifier.mjs ---------------------------
ALL_LABELS = ["rat", "ox", "tiger", "hare", "dragon", "snake", "horse", "ram",
              "monkey", "bird", "dog", "boar", "transition", "none"]
LABEL_TO_INDEX = {l: i for i, l in enumerate(ALL_LABELS)}
NUM_CLASSES = len(ALL_LABELS)
FEATURE_LENGTH = 135
NUM_LANDMARKS = 21
PER_HAND = NUM_LANDMARKS * 3
LEFT_OFF, RIGHT_OFF = 0, PER_HAND
PRESENCE_OFF = PER_HAND * 2
WRIST_DIST_OFF = PRESENCE_OFF + 2
REL_ROT_OFF = WRIST_DIST_OFF + 1
FINGERTIP_OFF = REL_ROT_OFF + 1

HIDDEN = (64, 32)
K_FOLDS = 5
SEED = 1337
NON_PERSON = "transitions-and-none"

# --- Load ---------------------------------------------------------------
try:
    from google.colab import files
    print("Select every file in datasets/raw/*.json (multi-select):")
    uploaded = files.upload()
    raw_files = {name: json.loads(content) for name, content in uploaded.items()}
except ImportError:
    # Local fallback, if you ever do get numpy/sklearn installed here.
    from pathlib import Path
    raw_dir = Path(__file__).resolve().parent.parent / "datasets" / "raw"
    raw_files = {p.name: json.loads(p.read_text()) for p in raw_dir.glob("*.json")}

labels, persons, features = [], [], []
for name, parsed in raw_files.items():
    if parsed.get("featureLength") != FEATURE_LENGTH:
        continue
    for s in parsed["samples"]:
        f = np.array(s["features"], dtype=np.float64)
        if f[PRESENCE_OFF] == 0 and f[PRESENCE_OFF + 1] == 0:
            continue  # no hand detected at all
        if np.any(np.abs(f) > 10):
            continue  # degenerate-hand-scale numerical blow-up, see features.ts MIN_HAND_SCALE
        if s["label"] not in LABEL_TO_INDEX:
            continue
        labels.append(LABEL_TO_INDEX[s["label"]])
        persons.append(s["person"])
        features.append(f)

X = np.array(features)
y = np.array(labels)
persons = np.array(persons)
print(f"loaded {len(X)} usable samples from {len(raw_files)} files")

# --- Mirror augmentation — see mirrorFeatures() in features.ts for the math
def mirror_features(F):
    out = np.zeros_like(F)
    for i in range(NUM_LANDMARKS):
        out[:, LEFT_OFF + i*3 + 0] = -F[:, RIGHT_OFF + i*3 + 0]
        out[:, LEFT_OFF + i*3 + 1] =  F[:, RIGHT_OFF + i*3 + 1]
        out[:, LEFT_OFF + i*3 + 2] =  F[:, RIGHT_OFF + i*3 + 2]
        out[:, RIGHT_OFF + i*3 + 0] = -F[:, LEFT_OFF + i*3 + 0]
        out[:, RIGHT_OFF + i*3 + 1] =  F[:, LEFT_OFF + i*3 + 1]
        out[:, RIGHT_OFF + i*3 + 2] =  F[:, LEFT_OFF + i*3 + 2]
    out[:, PRESENCE_OFF] = F[:, PRESENCE_OFF + 1]
    out[:, PRESENCE_OFF + 1] = F[:, PRESENCE_OFF]
    out[:, WRIST_DIST_OFF] = F[:, WRIST_DIST_OFF]
    out[:, REL_ROT_OFF] = -F[:, REL_ROT_OFF]
    out[:, FINGERTIP_OFF:FINGERTIP_OFF + 5] = F[:, FINGERTIP_OFF:FINGERTIP_OFF + 5]
    return out

def small_rotate(F, rng, max_radians=0.12):
    out = F.copy()
    angles = rng.uniform(-max_radians, max_radians, size=len(F))
    c, s = np.cos(angles), np.sin(angles)
    for base in (0, PER_HAND):
        for i in range(NUM_LANDMARKS):
            xi, yi = base + i*3, base + i*3 + 1
            x, y_ = out[:, xi].copy(), out[:, yi].copy()
            out[:, xi] = c*x - s*y_
            out[:, yi] = s*x + c*y_
    return out

def jitter(F, rng, sigma=0.015):
    return F + rng.uniform(-sigma, sigma, size=F.shape)

def augment(F, L, rng):
    mirrored = mirror_features(F)
    rotated = jitter(small_rotate(F, rng), rng)
    rotated_mirrored = jitter(small_rotate(mirrored, rng), rng)
    return np.vstack([F, mirrored, rotated, rotated_mirrored]), np.concatenate([L, L, L, L])

def make_classifier(seed):
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN, activation="relu", solver="adam",
        alpha=1e-4, learning_rate_init=0.01, max_iter=300,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        random_state=seed,
    )

def run_split(X_train, y_train, X_test, y_test, seed):
    rng = np.random.default_rng(seed)
    X_aug, y_aug = augment(X_train, y_train, rng)
    scaler = StandardScaler().fit(X_train)  # fit on un-augmented train only
    clf = make_classifier(seed).fit(scaler.transform(X_aug), y_aug)
    y_pred = clf.predict(scaler.transform(X_test)) if len(X_test) else np.array([])
    return clf, scaler, y_pred

# --- Phase 1: person-grouped K-fold cross-validation ------------------------
eligible = persons != NON_PERSON
gkf = GroupKFold(n_splits=K_FOLDS)
pooled_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
fold_accs = []

print(f"\n=== {K_FOLDS}-fold cross-validation (grouped by recorder) ===")
for i, (train_idx, test_idx) in enumerate(gkf.split(X[eligible], y[eligible], groups=persons[eligible])):
    # Map back to the full dataset: non-person (transitions/none session)
    # samples always join the training side, never held out.
    eligible_idx = np.where(eligible)[0]
    test_people = set(persons[eligible_idx[test_idx]])
    is_test = np.isin(persons, list(test_people)) & eligible
    X_tr, y_tr = X[~is_test], y[~is_test]
    X_te, y_te = X[is_test], y[is_test]
    print(f"\nfold {i+1}/{K_FOLDS} — held out: {', '.join(sorted(test_people))} ({len(X_te)} samples)")

    _, _, y_pred = run_split(X_tr, y_tr, X_te, y_te, SEED + i + 1)
    acc = (y_pred == y_te).mean()
    fold_accs.append(acc)
    print(f"  fold held-out accuracy: {acc*100:.1f}%")
    pooled_cm += confusion_matrix(y_te, y_pred, labels=range(NUM_CLASSES))

mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
print(f"\ncross-validated accuracy: {mean_acc*100:.1f}% ± {std_acc*100:.1f}% across {len(fold_accs)} folds")

print("\n=== pooled cross-validation results (all folds combined) ===")
support = pooled_cm.sum(axis=1)
present = support > 0
for c in range(NUM_CLASSES):
    if not present[c]:
        continue
    tp = pooled_cm[c, c]
    pred_col = pooled_cm[:, c].sum()
    prec = tp / pred_col if pred_col > 0 else float("nan")
    rec = tp / support[c]
    f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    print(f"  {ALL_LABELS[c]:<11} precision {prec*100:3.0f}%  recall {rec*100:3.0f}%  f1 {f1c*100:3.0f}%  (n={support[c]})")

print("\nconfusion matrix (rows=true, cols=predicted):")
header = "".join(f"{l[:4]:>6}" for l in ALL_LABELS)
print(" " * 12 + header)
for c in range(NUM_CLASSES):
    if not present[c]:
        continue
    print(f"{ALL_LABELS[c]:<12}" + "".join(f"{n:>6}" for n in pooled_cm[c]))

# --- Phase 2: final model on all data ----------------------------------------
print("\n=== training final model on all data ===")
clf, scaler, _ = run_split(X, y, np.empty((0, FEATURE_LENGTH)), np.empty((0,), dtype=int), SEED)

weights = {
    "version": 1,
    "labels": ALL_LABELS,
    "featureLength": FEATURE_LENGTH,
    "mean": scaler.mean_.tolist(),
    "std": scaler.scale_.tolist(),
    "hidden1": HIDDEN[0],
    "hidden2": HIDDEN[1],
    # sklearn's coefs_[k] shape is (in, out), row-major flatten == our i*out+j layout — matches classifier.ts exactly.
    "W1": clf.coefs_[0].flatten().tolist(),
    "b1": clf.intercepts_[0].tolist(),
    "W2": clf.coefs_[1].flatten().tolist(),
    "b2": clf.intercepts_[1].tolist(),
    "W3": clf.coefs_[2].flatten().tolist(),
    "b3": clf.intercepts_[2].tolist(),
}
with open("weights.json", "w") as f:
    json.dump(weights, f)
print(f"\nsaved weights.json (expected accuracy ~{mean_acc*100:.0f}% ± {std_acc*100:.0f}%, per cross-validation above)")

try:
    files.download("weights.json")
except NameError:
    print("(not in Colab — weights.json written to the current directory)")
