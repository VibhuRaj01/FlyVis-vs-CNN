# FlyVis MNIST — Detailed Project Report

This report explains what was built, why it was built that way, which
frameworks and data were used, and how to reproduce every result.

**Project root:** `/home/bibu/Desktop/Project/Flyvis`

**Goal:** train a digit classifier on MNIST using the pretrained FlyVis
connectome-constrained optic-lobe model, then compare it with a conventional
CNN baseline.

---

## 1. Executive summary

The repository started with two notebooks (`download_data.ipynb`,
`data_exploration.ipynb`) and a `data/` folder containing a FlyVis connectome,
a pretrained model, and response data. It did not contain a training script.

I added a complete MNIST pipeline:

1. map each MNIST image onto the FlyVis hexagonal photoreceptor lattice,
2. run the **frozen** pretrained FlyVis network for a short static-image
   sequence,
3. pool its cell-type activity into a compact feature vector,
4. train a small MLP readout on those features,
5. train a small CNN baseline on raw pixels with the identical data split.

Final full-MNIST results:

| Model | Validation | Test |
| --- | ---: | ---: |
| Frozen FlyVis + MLP readout | 98.70% | **98.26%** |
| Small CNN baseline | 99.58% | **99.55%** |

The CNN is about 1.3 percentage points better. The FlyVis result is still
notable because the biological network was never trained for digit
classification and only a small readout was fitted on top of its frozen
activity.

---

## 2. Original repository state

The original directory contained:

```text
Flyvis/
├── .env
├── data/
│   ├── connectome/ConnectomeFromAvgFilters_0000/
│   ├── results/flow/0000/
│   ├── responses_norm.h5
│   ├── results_pretrained_models.zip
│   └── results_umap_and_clustering.zip
├── download_data.ipynb
└── data_exploration.ipynb
```

The `.env` file contained:

```text
FLYVIS_ROOT_DIR=/home/bibu/Desktop/Project/Flyvis/data
```

The notebooks did two things:

- `download_data.ipynb` ran `flyvis download-pretrained`, copied package data
  into `data/`, wrote `.env`, and extracted the pretrained-model and
  clustering zips.
- `data_exploration.ipynb` inspected `responses_norm.h5`, including the
  `norm` and `rectified_norm` response matrices, and plotted response norms by
  cell type and model.

There was no MNIST training code, no test suite, no linter config, and no CI.

---

## 3. Frameworks and environment

All training and evaluation used the existing virtual environment at
`/home/bibu/flyvis`, not the system Python.

| Component | Version / detail |
| --- | --- |
| Python | 3.12.15 |
| Operating system | Linux 6.18.56 (x86_64, glibc 2.44) |
| FlyVis | 1.2.0 |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.1+cu130 |
| NumPy | 2.5.3 |
| CUDA build | CUDA 13.0 |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB |
| Matplotlib | 3.11.2 (used by the existing notebooks) |

**Important:** system `python3` does not have FlyVis installed. The exact
interpreter is:

```bash
/home/bibu/flyvis/bin/python
```

FlyVis itself depends on internal packages such as `datamate`, `h5py`, `toolz`,
and `joblib`; these were already installed and were not modified.

### Environment variables

Set before importing FlyVis/torch:

```python
os.environ.setdefault("FLYVIS_ROOT_DIR", str(ROOT / "data"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

`FLYVIS_ROOT_DIR` is required because FlyVis looks there for `results/` and
`connectome/`. FlyVis loads `.env` from the **current working directory** only,
so scripts are intended to run from the repository root. `train_mnist.py`
also sets the variable itself so it works when launched elsewhere.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces CUDA memory
fragmentation during the relatively large network simulation.

---

## 4. Data

### 4.1 MNIST

MNIST is downloaded through `torchvision.datasets.MNIST` into:

```text
.data/MNIST/
```

Both scripts use the standard 60,000-image training set and 10,000-image test
set. The training pool is split into 55,000 train and 5,000 validation images
with seed 0; the official 10,000-image test set is used for the final report.

### 4.2 FlyVis connectome and pretrained model

Required paths:

```text
data/connectome/ConnectomeFromAvgFilters_0000/
data/results/flow/0000/000/
```

The pretrained directory contains:

```text
_meta.yaml
best_chkpt
chkpts/chkpt_00000
validation/loss.h5
```

The model is loaded with:

```python
from flyvis import NetworkView

network_view = NetworkView("flow/0000/000")
net = network_view.init_network(checkpoint="best")
```

Relevant sizes and structure:

| Property | Value |
| --- | ---: |
| Nodes | 45,669 |
| Edges | 1,513,231 |
| Cell types | 65 |
| Photoreceptor input types | 8 (`R1`–`R8`) |
| Hexal positions per input type | 721 |
| Output cell types | 34 |
| Free network parameters | 734 |
| Fixed network parameters | 2,959 |

The pretrained model is a `PPNeuronIGRSynapses` dynamic network trained on the
FlyVis `flow` task with Sintel data. It is a mechanistic optic-flow model, not
a conventional image classifier.

For MNIST, the network is kept **frozen**. Only the readout is trained. This
keeps the pretrained optic-flow representation intact and avoids
backpropagating through a 45,669-node, 1.5-million-edge simulation.

---

## 5. Problem framing

“Train FlyVis on MNIST” can mean several things:

1. **End-to-end fine-tuning:** train the connectome network and a classifier
   jointly. This is expensive and can destroy the pretrained optic-flow
   behavior.
2. **Frozen feature extraction:** run the pretrained network on MNIST and train
   a small classifier on its activity.
3. **Replace the optic-flow decoder with a digit decoder:** a variant of (2).

I chose (2), which is the standard and practical way to probe what a frozen
pretrained representation contains. It is fast, reproducible on a single GPU,
and isolates the representation quality from the readout capacity.

The phrase “FlyVis classifier” below therefore means:

> frozen pretrained FlyVis simulation → pooled activity features → trained MLP
> readout

It does **not** mean the 45k-node network itself was trained on MNIST.

---

## 6. End-to-end FlyVis pipeline

The implementation is in `train_mnist.py`.

### 6.1 Mapping MNIST onto the FlyVis lattice

FlyVis photoreceptors live on a hexagonal lattice in axial coordinates
`(u, v)` with extent 15. The input cell type `R1` has 721 positions with
`u, v ∈ [-15, 15]`.

Each 28×28 MNIST image is bilinearly sampled at those 721 positions using
`torch.nn.functional.grid_sample`. The normalized sampling coordinates are:

```python
gx = 2 * (u - u.min()) / (u.max() - u.min()) - 1
gy = 2 * (v - v.min()) / (v.max() - v.min()) - 1
```

Then:

```python
sampled = F.grid_sample(
    images,
    grid.expand(batch, -1, -1, -1),
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
)
hex_input = sampled[:, 0, 0]  # (batch, 721)
```

The result is one 721-element “retina image” per MNIST digit.

### 6.2 Broadcasting to the eight photoreceptor types

FlyVis has eight input cell types (`R1`–`R8`). `Stimulus.add_input` indexes a
buffer of shape `(samples, frames, nodes)` with an input index of shape
`(8, 721)`. Passing a tensor of shape `(batch, frames, 1, 721)` therefore
broadcasts the same image to all eight photoreceptor types.

The network input shape is:

```text
(batch, frames, 1, 721)
```

not `(batch, frames, 1, 8 * 721)`.

### 6.3 Static-image simulation

MNIST images have no temporal dimension, so each image is repeated for six
frames:

```python
movie = hex_input[:, None, None, :].repeat(1, n_frames, 1, 1)
```

Simulation parameters:

```python
n_frames = 6
dt = 0.02
```

`dt = 0.02` is the largest value FlyVis recommends without a warning
(`dt <= 1/50`).

Calling `net.simulate(..., initial_state="auto")` would compute a 1-second
grey-input steady state (50 integration steps) for every batch. That dominated
runtime. Instead, the steady state is computed once per batch size and cached:

```python
initial_state = net.steady_state(1.0, dt, batch_size)
activity = net.simulate(movie, dt, initial_state=initial_state)
```

This reduced feature extraction from roughly 15 images/second to roughly
1,000–1,200 images/second.

The returned activity has shape:

```text
(batch, 6, 45,669)
```

The last three frames are averaged to reduce transient/noise effects:

```python
late = activity[:, -3:, :].mean(dim=1)  # (batch, 45,669)
```

### 6.4 Feature extraction from cell-type activity

The model has 65 cell types. The eight photoreceptor types are excluded (they
would mostly duplicate the input image), leaving 57 cell types.

For each cell type:

1. Select its node indices from `connectome.nodes.layer_index`.
2. Read the matching `(u, v)` coordinates from `connectome.nodes.u/v`.
3. Scatter activity into a dense 31×31 axial map (`u, v ∈ [-15, 15]`).
4. Apply 7×7 adaptive mean and max pooling.
5. For mean pooling, divide by the occupancy count so empty lattice positions
   do not dilute the average.

Per cell type this gives `7 × 7 × 2 = 98` features. Across 57 cell types:

```text
57 × 49 × 2 = 5,586 features
```

Most cell types have 721 nodes. Two exceptions are `Lawf1` and `Lawf2`, which
have only 123 nodes. The scatter maps are therefore built per cell type rather
than assuming 721 nodes everywhere.

### 6.5 Feature caching

Extraction is deterministic and relatively expensive, so features are saved as
float16 tensors:

```text
runs/flyvis_mnist/cache/train_n60000_seed0_f6_p7_dt0.02_non-input.pt
runs/flyvis_mnist/cache/test_n10000_seed0_f6_p7_dt0.02_non-input.pt
```

The cache key includes:

- split (`train` / `test`),
- number of images,
- seed,
- number of frames,
- pool size,
- `dt`,
- cell-type selection.

That prevents silently reusing features produced with different settings.
`--no-cache` forces re-extraction.

Full-MNIST cache size is about 750 MB (640 MB train, 107 MB test). Deleting it
only forces re-extraction.

### 6.6 Train/validation/test split

For the training pool:

```python
pool = randperm(60_000, seed=0)[:train_size]
order = randperm(len(pool), seed=0)
val_idx = pool[order[:5_000]]
train_idx = pool[order[5_000:]]
```

For the test pool:

```python
test_idx = randperm(10_000, seed=0)[:test_size]
```

The CNN baseline uses the exact same split and seed. This makes the two test
accuracies directly comparable.

### 6.7 Readout model and training

Features are standardized using training-set mean and standard deviation only.

Readout architecture:

```text
5,586 → Linear → 128 → ReLU → Dropout(0.2) → Linear → 10
```

Training configuration:

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-2` |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping | patience 10 on validation accuracy |
| Parameters | 716,426 |
| Best epoch | 24 |

The checkpoint with the best validation accuracy is restored before the final
train/validation/test evaluation.

The script reseeds immediately before readout construction so the readout
initialization and batch order do not depend on whether features were loaded
from cache or extracted in the current run.

### 6.8 Evaluation and artifacts

Saved to `runs/flyvis_mnist/`:

| File | Contents |
| --- | --- |
| `readout.pt` | readout weights, feature mean/std, config |
| `metrics.json` | accuracy, history, confusion matrix, timings |
| `test_predictions.pt` | test predictions and labels |

---

## 7. CNN baseline

The implementation is in `train_cnn.py`. It does not import FlyVis.

### 7.1 Architecture

```text
Conv2d(1 → 32, 3×3) → BatchNorm → ReLU
Conv2d(32 → 32, 3×3) → ReLU → MaxPool(2)
Conv2d(32 → 64, 3×3) → BatchNorm → ReLU
Conv2d(64 → 64, 3×3) → ReLU → MaxPool(2)
Flatten
Linear(64×7×7 → 128) → ReLU → Dropout(0.5)
Linear(128 → 10)
```

Parameters: 468,010.

### 7.2 Training

| Setting | Value |
| --- | --- |
| Input | 28×28 raw pixels |
| Normalization | mean 0.1307, std 0.3081 |
| Augmentation | none |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 128 |
| LR schedule | cosine annealing |
| Max epochs | 20 |
| Early stopping | patience 3 on validation accuracy |
| Best epoch | 18 |

No augmentation was used so the comparison stays close to the FlyVis pipeline,
which also has no digit-specific augmentation.

---

## 8. Results

### 8.1 Overall accuracy

| Model | Val | Test | Train | Best epoch | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen FlyVis + MLP | 98.70% | **98.26%** | 99.63% | 24 | 716,426 |
| Small CNN | 99.58% | **99.55%** | 100.00%* | 18 | 468,010 |

\*Reported train accuracy is 99.99% due to the final partial training batch.

### 8.2 Per-class test accuracy

| Digit | FlyVis + MLP | CNN |
| ---: | ---: | ---: |
| 0 | 98.47% | 99.80% |
| 1 | 99.30% | 99.91% |
| 2 | 97.87% | 99.71% |
| 3 | 98.71% | 99.70% |
| 4 | 98.07% | 99.69% |
| 5 | 97.53% | 99.44% |
| 6 | 99.06% | 99.37% |
| 7 | 98.05% | 99.22% |
| 8 | 98.05% | 99.49% |
| 9 | 97.32% | 99.11% |

The FlyVis readout is weakest on 5, 9, 2, and 4. The CNN is above 99% on every
digit.

### 8.3 Interpretation

The CNN wins because it learns filters directly from static digit pixels. FlyVis
was trained on optic-flow movies and its activity reflects motion-sensitive
circuits. A frozen optic-flow representation still separates digits surprisingly
well (98.26%), but it is not specialized for handwritten digit shapes.

This is a meaningful result: it shows the pretrained biological network
contains generic visual information that transfers to a completely different
task, while also showing the expected gap to a task-trained CNN.

---

## 9. Experiments and hyperparameter tuning

Initial probes on 5,000 training images and 1,000 test images:

- linear readout: about 95.4%,
- MLP readout: about 94.4% (overfit at that data size).

After extracting full-MNIST features, I compared readout configurations on the
validation split:

| Hidden | Dropout | Weight decay | Val | Test |
| ---: | ---: | ---: | ---: | ---: |
| 0 (linear) | — | `1e-2` | 98.22% | 98.16% |
| 128 | 0.2 | `1e-2` | **98.76%** | **98.52%** |
| 256 | 0.2 | `1e-2` | 98.42% | 98.27% |
| 512 | 0.2 | `1e-2` | 98.20% | 97.99% |
| 512 | 0.4 | `1e-2` | 98.52% | 98.40% |
| 1024 | 0.5 | `1e-2` | 98.66% | 98.29% |
| 512 | 0.3 | `1e-3` | 98.74% | 98.46% |
| 256 | 0.5 | `1e-3` | 98.74% | 98.53% |

The final default is `hidden=128`, `dropout=0.2`, `weight_decay=1e-2`. The
number in `runs/flyvis_mnist/metrics.json` is authoritative; small differences
from the table are expected because re-extracted features differ by tiny
float16 rounding amounts, which can slightly change the training trajectory.

The final reported run used 55,000 training images and 5,000 validation images.

---

## 10. Performance and memory

On the RTX 4080 Laptop GPU:

| Stage | Time |
| --- | ---: |
| FlyVis feature extraction, 60k train | ~49 s |
| FlyVis feature extraction, 10k test | ~8 s |
| Readout training (cached features) | ~5 s |
| CNN baseline, full run | ~33 s |

Key performance choices:

- Precompute `steady_state` once per batch size.
- Use batch size 32 for FlyVis simulation.
- Cache features as float16.
- Train the readout with batch size 256 on the GPU.

Memory notes:

- Batch size 32 is safe on the 12 GB GPU.
- Batch size 64 caused CUDA out-of-memory in `steady_state`, because that path
  materializes a long sequence of full network states.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps avoid fragmentation.

---

## 11. Bugs, pitfalls, and fixes

These were the non-obvious issues encountered while building the pipeline.

### 11.1 FlyVis changes the global default torch device

Importing FlyVis calls:

```python
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
```

Consequences:

- `torch.Generator()` defaults to CPU, but `torch.randperm(...)` may default to
  CUDA, causing “Expected a 'cuda' device type for generator but found 'cpu'”.
- HDF5 access such as `nodes.u[idx]` fails if `idx` is a CUDA tensor.
- Newly created tensors silently land on the GPU.

Fixes:

- Pass `device="cpu"` explicitly to generators and `randperm`.
- Build HDF5 indices with `np.asarray(layer_index[name])` and index with NumPy
  integers, or move tensors with `.cpu().numpy()`.

### 11.2 Byte strings vs. ordinary strings

`connectome.nodes.layer_index` keys are ordinary strings, but
`input_cell_types` and `output_cell_types` contain byte strings such as
`b"R1"`. They must be decoded before dictionary lookups.

### 11.3 Cell types do not all have 721 nodes

`Lawf1` and `Lawf2` have 123 nodes. The feature map code reads each type's own
coordinates and builds its own occupancy mask.

### 11.4 Dropout was active during confusion-matrix computation

An early version of `evaluate()` always left the model in training mode. The
final confusion matrix was then computed with dropout active, making it
inconsistent with the reported accuracy. Fixed by saving/restoring the previous
mode in `evaluate()` and calling `model.eval()` before the final prediction
pass.

### 11.5 Cache key did not include `dt`

Changing `dt` would have reused stale features. The cache filename now includes
`dt`.

### 11.6 Reproducibility across cached and uncached runs

Feature extraction and readout training originally consumed different amounts
of global RNG depending on whether the cache was hit. The script now reseeds
immediately before constructing the readout.

### 11.7 `simulate()` disables gradients

`Network.simulate()` wraps the forward pass in FlyVis's `simulation()` context,
which sets all parameters to `requires_grad=False`. It is for inference only.
End-to-end fine-tuning would need `network.forward()` under
`network.enable_grad(True)`, not `simulate()`.

### 11.8 Slow repeated steady-state warmup

Without a precomputed initial state, every batch repeats 50 integration steps
of grey input. Caching `steady_state` per batch size is the single biggest
speedup in the pipeline.

---

## 12. Files created or changed

| Path | Purpose |
| --- | --- |
| `train_mnist.py` | Frozen FlyVis feature extraction + MLP readout training |
| `train_cnn.py` | CNN baseline on raw MNIST pixels |
| `README.md` | Short project overview, commands, results, conclusion |
| `AGENTS.md` / `AGENT.md` | Local coding-agent guidance (git-ignored) |
| `PROJECT_REPORT.md` | This detailed report |
| `.gitignore` | Ignores `data/`, `.data/`, `runs/`, agent files, and Python caches |
| `runs/flyvis_mnist/` | FlyVis readout, metrics, predictions, feature cache |
| `runs/cnn_mnist/` | CNN checkpoint, metrics, predictions |
| `.data/MNIST/` | Downloaded MNIST data |

The original `data/` contents and the two existing notebooks were not modified.
`data/`, `.data/`, `runs/`, `AGENTS.md`, and `AGENT.md` are intentionally
git-ignored, so the pushed repository contains code, docs, and notebooks only.

---

## 13. Reproduction

From the repository root, using the exact FlyVis interpreter:

```bash
cd /home/bibu/Desktop/Project/Flyvis

# Check the environment and data root
/home/bibu/flyvis/bin/python -c "import flyvis; print(flyvis.__version__, flyvis.root_dir)"

# Full FlyVis pipeline
/home/bibu/flyvis/bin/python train_mnist.py

# CNN baseline
/home/bibu/flyvis/bin/python train_cnn.py
```

Quick smoke test:

```bash
/home/bibu/flyvis/bin/python train_mnist.py \
  --train-size 2000 --test-size 1000 --epochs 3 --no-cache \
  --cache-dir /tmp/flyvis_smoke_cache --out-dir /tmp/flyvis_smoke_out
```

Force feature re-extraction:

```bash
/home/bibu/flyvis/bin/python train_mnist.py --no-cache
```

Useful `train_mnist.py` options:

| Option | Meaning |
| --- | --- |
| `--train-size` | Number of MNIST training images |
| `--test-size` | Number of MNIST test images |
| `--val-size` | Validation images taken from the training pool |
| `--n-frames` | Number of repeated static frames |
| `--dt` | Integration step (`<= 0.02`) |
| `--pool` | Spatial pool size per cell type |
| `--cell-types` | `non-input`, `output`, or `all` |
| `--hidden` | Readout hidden size (`0` = linear) |
| `--dropout` | Readout dropout |
| `--epochs` | Maximum readout epochs |
| `--patience` | Early-stopping patience |
| `--no-cache` | Recompute features |

---

## 14. Limitations

- The FlyVis network is frozen; only the readout is trained.
- MNIST images are static and repeated for six frames. This does not match the
  moving-image optic-flow distribution FlyVis was trained on.
- The readout is a small MLP on coarse 7×7 per-cell-type maps. A stronger
  readout could improve accuracy.
- No digit-specific augmentation was used.
- The CNN uses standard MNIST normalization while FlyVis uses raw `[0, 1]`
  values mapped onto the hex lattice.
- The comparison is on a single train/validation/test split and seed.

---

## 15. Possible next steps

1. **End-to-end fine-tuning:** unfreeze the 734 FlyVis free parameters and
   backpropagate through `network.forward()` with `enable_grad`. This is much
   more expensive and risks degrading the pretrained optic-flow behavior.
2. **Moving MNIST:** feed translating digits to better match the temporal
   optic-flow training distribution.
3. **Stronger readout:** use a small CNN over the 57×31×31 activity maps instead
   of mean/max pooling.
4. **Augmentation:** random shifts/rotations for the CNN baseline and possibly
   the hex-mapped inputs.
5. **Ensemble:** average multiple readouts or multiple pretrained FlyVis model
   instances.

---

## 16. Conclusion

The project now has a complete, reproducible MNIST pipeline built around the
pretrained FlyVis model, a matched CNN baseline, cached features, saved
checkpoints, metrics, and documentation.

The frozen FlyVis representation reaches **98.26%** test accuracy, while a small
CNN reaches **99.55%**. The gap is expected: FlyVis is a frozen optic-flow model,
not a digit classifier. The result demonstrates that a connectome-constrained
biological visual system contains transferable visual features, while also
showing the advantage of task-specific end-to-end learning.
