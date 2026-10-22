# FlyVis MNIST: Biological Vision vs. Convolutional Neural Networks

Classify handwritten digits from the MNIST dataset using a **pretrained, connectome-constrained FlyVis neural network** and compare its performance against a conventional Convolutional Neural Network (CNN).

This project explores whether a biologically inspired visual processing network, originally designed for optic-flow modeling, can learn to distinguish handwritten digits without being trained specifically for classification.

The FlyVis network remains **frozen during training**. Only a lightweight multilayer perceptron (MLP) readout is trained on its neural activity, allowing the experiment to run efficiently on a single GPU.

## Key Results

| Model | Input Representation | Trainable Parameters | Validation Accuracy | Test Accuracy |
|---|---|---:|---:|---:|
| Frozen FlyVis + MLP | 5,586 neural activity features | 716,426 | 98.70% | 98.26% |
| Small CNN baseline | Raw 28 × 28 pixels | 468,010 | 99.58% | **99.55%** |

Both models use the same training, validation, and test splits with random seed `0`.

The CNN outperforms the frozen FlyVis model by approximately **1.29 percentage points** on the test set. However, FlyVis achieves 98.26% accuracy despite being pretrained for optic-flow modeling rather than digit classification.

## Project Overview

FlyVis is a biologically inspired model of the fruit fly's visual system. Its network architecture is constrained by the organization of the fly optic-lobe connectome.

This project uses the pretrained FlyVis network as a fixed feature extractor:

- **Biological vision:** Process images through the pretrained FlyVis network.
- **Neural feature extraction:** Convert simulated neural activity into compact feature vectors.
- **Classification:** Train an MLP to classify the extracted features into ten digit classes.
- **Baseline comparison:** Evaluate a conventional CNN trained directly on MNIST pixels.

The experiment provides a controlled starting point for investigating how biological visual representations compare with task-specific deep learning models.

## Architecture

<pre>
                  MNIST Image
                   28 × 28
                       |
              Bilinear Sampling
                       |
             FlyVis Hexagonal Lattice
               721 Photoreceptors
                       |
                       v
             Pretrained FlyVis Network
               Frozen Parameters
                       |
             Neural Activity Extraction
                       |
           Spatial Pooling Across Cell Types
                       |
              5,586-Dimensional Vector
                       |
                       v
                 MLP Readout
                 Trainable
                       |
                       v
                10 Digit Classes
                 0, 1, ..., 9
</pre>

The CNN baseline follows an independent pipeline:

```text
MNIST Image (28 × 28)
        |
        v
  Four-Convolution CNN
        |
        v
   Digit Classification
```

## How It Works

### 1. Map MNIST Images to the FlyVis Retina

Each 28 × 28 MNIST image is bilinearly sampled at the 721 photoreceptor positions of the FlyVis hexagonal lattice.

This maps the standard image representation onto the spatial arrangement expected by the pretrained visual network.

### 2. Run the Pretrained Network

Each image is repeated across six simulation frames using `dt = 0.02`.

The pretrained FlyVis network processes these frames, generating neural activity throughout its network of approximately 45,669 nodes.

The FlyVis parameters remain frozen throughout the experiment.

### 3. Extract Neural Features

For each non-photoreceptor cell type:

- Extract activity from the final three simulation frames.
- Spatially pool the activity into a 7 × 7 grid.
- Calculate both mean and maximum pooled activity.

Combining the pooled representations produces a **5,586-dimensional feature vector** for each image.

### 4. Train the Classification Readout

A one-hidden-layer MLP is trained on the extracted neural features.

The dataset is divided into:

| Split | Number of Images |
|---|---:|
| Training | 55,000 |
| Validation | 5,000 |
| Testing | 10,000 |

The validation set is used during model development, while the official MNIST test set is reserved for final evaluation.

### 5. Compare Against a CNN

The independent CNN baseline trains directly on normalized 28 × 28 MNIST images.

It uses the same data split and random seed, making the results easier to compare under consistent experimental conditions.

## Requirements

- Python environment with FlyVis and its required dependencies installed
- PyTorch and torchvision
- NumPy and the other dependencies required by the training scripts
- Pretrained FlyVis connectome data and checkpoint
- GPU recommended for faster feature extraction and training

The experiment was tested on an **NVIDIA RTX 4080 Laptop GPU with 12 GB of VRAM**.

Exact execution time and memory requirements may vary with hardware, software versions, and configuration.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your GitHub repository URL and directory name.

### 2. Set Up the Python Environment

Activate your existing FlyVis environment, or create a suitable environment following the installation instructions for FlyVis.

Verify that FlyVis is installed:

```bash
python -c "import flyvis; print(flyvis.__version__, flyvis.root_dir)"
```

Use the Python interpreter associated with your FlyVis environment when running the scripts.

### 3. Configure FlyVis Data

Configure `FLYVIS_ROOT_DIR` to point to the directory containing the FlyVis data.

The directory should contain the required connectome and pretrained checkpoint, including the `flow/0000/000` checkpoint used in this experiment.

A local `.env` file can be used for environment-specific configuration. Do not commit machine-specific paths or private configuration files.

### 4. Prepare the Datasets

MNIST is downloaded automatically into `.data/MNIST` when required.

If the FlyVis data has not been downloaded, run:

```text
download_data.ipynb
```

Ensure that the pretrained checkpoint and connectome data are available before starting training.

## Running the Experiments

### FlyVis-Based Classifier

Run the complete MNIST experiment:

```bash
python train_mnist.py
```

Run a smaller experiment for a quick sanity check:

```bash
python train_mnist.py \
  --train-size 2000 \
  --test-size 1000 \
  --epochs 3 \
  --no-cache
```

Feature extraction is cached under `runs/flyvis_mnist/cache/`. Subsequent runs can reuse cached features when the cache is valid.

Use `--no-cache` to recompute the features.

### CNN Baseline

Train the CNN directly on MNIST images:

```bash
python train_cnn.py
```

The CNN baseline uses the same data split and seed as the FlyVis experiment.

## Outputs

Training generates the following artifacts.

### FlyVis Model

```text
runs/
└── flyvis_mnist/
    ├── cache/
    ├── readout.pt
    ├── metrics.json
    └── test_predictions.pt
```

- `cache/` — cached neural features used during readout training.
- `readout.pt` — trained readout weights, feature normalization parameters, and configuration.
- `metrics.json` — accuracy, training history, and confusion matrix.
- `test_predictions.pt` — test predictions and corresponding labels.

### CNN Baseline

```text
runs/
└── cnn_mnist/
    ├── ...
```

The CNN output directory contains the corresponding evaluation artifacts.

Generated datasets, checkpoints, caches, and experiment outputs should generally remain local and be excluded from Git unless they are intentionally published.

## Results and Interpretation

The frozen FlyVis representation achieves **98.26% test accuracy**, demonstrating that the pretrained network's activity contains useful information for distinguishing handwritten digits.

However, the CNN achieves **99.55% test accuracy** because it learns filters directly from the classification task, whereas the FlyVis network is optimized for a different visual-processing objective and cannot adapt its internal parameters in this experiment.

The reported runtime on the RTX 4080 Laptop GPU was approximately:

| Experiment Stage | Runtime |
|---|---:|
| FlyVis feature extraction | 49 seconds |
| MLP readout training | 5 seconds |
| CNN training and evaluation | 33 seconds total |

These are observed runtimes from the reported experiment, not guaranteed benchmarks.

The results suggest that a connectome-constrained visual representation can transfer effectively to a static image classification task, even without task-specific training of the biological network.

## Limitations

- **Frozen biological network:** FlyVis parameters are not fine-tuned for digit classification.
- **Static stimuli:** Each digit is repeated over six frames rather than presented as a moving stimulus.
- **Task mismatch:** The pretrained network models optic-flow-related visual processing, while MNIST classification requires static shape recognition.
- **Different input pipelines:** FlyVis receives pixel values in the `[0, 1]` range mapped onto a hexagonal lattice; the CNN uses standard MNIST normalization.
- **Architecture differences:** The models use different representations and learning procedures, so this is a practical baseline comparison rather than a perfectly controlled comparison of biological and artificial learning.
- **Reproducibility:** Results and runtimes can vary with random seeds, hyperparameters, dependency versions, and feature extraction settings.

## Future Work

Potential directions for extending this project include:

1. **Moving-digit classification:** Introduce temporal stimuli that better match FlyVis's optic-flow training distribution.
2. **End-to-end fine-tuning:** Investigate whether adapting trainable FlyVis parameters improves classification performance.
3. **Readout experiments:** Compare linear classifiers, MLPs, and alternative pooling strategies.
4. **Feature analysis:** Identify which cell types contribute most to classification accuracy.
5. **Representation visualization:** Visualize neural activity, feature embeddings, and confusion matrices.
6. **Additional datasets:** Extend the comparison to more complex image classification tasks.
7. **Attention-based models:** Explore whether attention mechanisms can improve readout performance or capture interactions between biological cell types.

## Repository Structure

```text
.
├── train_mnist.py
├── train_cnn.py
├── download_data.ipynb
├── data_exploration.ipynb
├── PROJECT_REPORT.md
├── README.md
├── .env.example
├── .gitignore
└── runs/
```

The structure above describes the main project components. Generated directories and optional configuration files may differ depending on your local setup.

| File | Purpose |
|---|---|
| `train_mnist.py` | FlyVis feature extraction and MLP readout training |
| `train_cnn.py` | CNN baseline trained on raw MNIST images |
| `download_data.ipynb` | FlyVis data download and extraction |
| `data_exploration.ipynb` | Exploration of neural response norms |
| `PROJECT_REPORT.md` | Detailed experimental report |

## Acknowledgments

This project builds on the pretrained FlyVis visual-system model and the MNIST handwritten-digit dataset.

Please consult the original FlyVis project and its associated publications for model provenance, dataset details, and applicable licensing requirements.

## Conclusion

The experiment shows that a pretrained, connectome-constrained FlyVis network can support highly accurate MNIST classification using a learned readout alone.

Although the conventional CNN performs better on static handwritten digits, the FlyVis model offers an interesting starting point for studying biological visual representations, transfer learning, and the relationship between neural circuitry and machine learning.

**The central research question:** How much task-relevant information can a biologically constrained visual system provide without being retrained for the task?
