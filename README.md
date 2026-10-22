# FlyVis MNIST

Classify MNIST digits with the pretrained **FlyVis** connectome-constrained
optic-lobe network. The FlyVis simulation is frozen; only a small readout head
is trained on its pooled activity. This makes the experiment fast enough to run
on a single 12 GB GPU while preserving the pretrained optic-flow representation.

## Environment

Use the existing FlyVis virtual environment from the repository root:

```bash
cd /home/bibu/Desktop/Project/Flyvis
/home/bibu/flyvis/bin/python -c "import flyvis; print(flyvis.__version__, flyvis.root_dir)"
```

`.env` points `FLYVIS_ROOT_DIR` at `data/`, which contains the connectome and the
pretrained `flow/0000/000` checkpoint. MNIST is downloaded automatically into
`.data/MNIST`.

## Run

```bash
# Full MNIST (60,000 train / 10,000 test)
/home/bibu/flyvis/bin/python train_mnist.py

# Quick check
/home/bibu/flyvis/bin/python train_mnist.py \
  --train-size 2000 --test-size 1000 --epochs 3 --no-cache

# CNN baseline on raw pixels (same split and seed)
/home/bibu/flyvis/bin/python train_cnn.py
```

Feature extraction is cached in `runs/flyvis_mnist/cache/`, so changing readout
hyperparameters does not require rerunning FlyVis. Use `--no-cache` to recompute.

Training writes:

- `runs/flyvis_mnist/readout.pt` — readout weights, feature normalization, config
- `runs/flyvis_mnist/metrics.json` — accuracy, history, confusion matrix
- `runs/flyvis_mnist/test_predictions.pt` — test predictions and labels
- `runs/cnn_mnist/` — the same artifacts for the CNN baseline

## Results

Both models use the same 55k train / 5k validation / 10k test split and seed 0.
The test column is the official MNIST test set.

| Model | Input representation | Trainable parameters | Val | Test |
| --- | --- | ---: | ---: | ---: |
| FlyVis frozen + MLP readout | 57 cell types → 5,586 pooled features | 716,426 | 98.70% | **98.26%** |
| Small CNN baseline | raw 28×28 pixels, standard normalization | 468,010 | 99.58% | **99.55%** |

The CNN is about 1.3 percentage points more accurate. That is expected: it is
trained end-to-end on static digits, while FlyVis is a frozen optic-flow model
whose 45,669-node activity is only read out by a small MLP. The FlyVis run needs
about 49 s for feature extraction plus 5 s for readout training; the CNN takes
about 33 s total on the RTX 4080 Laptop GPU. Accuracy can shift slightly if
features are re-extracted or the seed/hyperparameters change; each
`metrics.json` records the exact run.

## How it works

1. Each 28x28 MNIST image is bilinearly sampled onto the 721 photoreceptor
   positions of the FlyVis hexagonal lattice.
2. The image is repeated for six frames (`dt=0.02`) and passed through the
   frozen pretrained network.
3. For every non-photoreceptor cell type, activity from the last three frames is
   spatially pooled into a 7x7 map (mean and max), producing a 5,586-dimensional
   feature vector.
4. A one-hidden-layer MLP is trained on those features with a train/validation
   split from the 60k training set; the 10k test set is used only for the final
   report.

`train_cnn.py` is an independent baseline: a small four-convolution CNN trained
on raw 28×28 pixels with the same split and seed.

## Files

- `train_mnist.py` — FlyVis feature extractor and readout training
- `train_cnn.py` — small CNN baseline on raw pixels
- `download_data.ipynb` — one-time FlyVis data download/extraction
- `data_exploration.ipynb` — response-norm exploration
- `PROJECT_REPORT.md` — detailed explanation of the pipeline, experiments, and results

Datasets under `data/` and `.data/` are local-only and git-ignored; run
`download_data.ipynb` to regenerate the FlyVis data. Agent guidance files are
also kept local and are not committed.

## Conclusion

The frozen FlyVis representation is competitive but not better than a task-trained
CNN on static MNIST: 98.26% versus 99.55% test accuracy. The result is useful as a
biological sanity check: a 5,586-dimensional readout of a connectome-constrained
optic-flow model already separates digits well, even though the network was never
trained for classification. The CNN remains the stronger baseline because it can
learn digit-specific filters end-to-end.

Natural next steps would be end-to-end fine-tuning of the FlyVis free parameters
(much more expensive), temporal/moving-digit stimuli that better match the optic-flow
training distribution, or a stronger readout / larger feature set.

## Limitations

- The readout is trained; the FlyVis parameters are not fine-tuned.
- MNIST images are static, repeated frames. This is a classification demo, not
  the temporal optic-flow task FlyVis was trained on.
- The CNN uses standard MNIST normalization; FlyVis uses raw `[0, 1]` pixel
  values mapped onto the hex lattice.
