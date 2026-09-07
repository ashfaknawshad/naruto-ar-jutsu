# Raw dataset contributions

One JSON file per recording session, exactly as downloaded from the
recorder tool (`?mode=record`) — don't edit or merge them by hand.

**Naming:** the recorder already names the download
`kekkai-dataset-<name>-<timestamp>.json` — just drop it in here unchanged.

**Do not commit:**
- Video or images. The recorder never captures them; only landmark
  coordinates leave your machine.
- Partial/test sessions — use Undo (Backspace) in the recorder to discard a
  bad take before downloading, rather than uploading a file you plan to hand-edit.

Each file's samples are tagged with the recorder's name, which is how the
training script later splits people into train vs. held-out test — so one
file per person per session, not merged across people.
