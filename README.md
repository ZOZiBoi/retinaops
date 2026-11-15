# RetinaOps

An end-to-end MLOps pipeline for 5-class diabetic retinopathy (DR) grading on fundus images.

The focus is a reproducible pipeline, not state-of-the-art accuracy. Phase 1 covers data
versioning, a repeatable training pipeline, experiment tracking and a model registry.

| Grade | Class | Images | Share |
|---|---|---|---|
| 0 | No DR | 1805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative DR | 295 | 8.1% |

The data is the APTOS 2019 Blindness Detection set (3,662 images, 224×224 pixels,
[Kaggle mirror](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)).

## Phase 1 at a glance

| Requirement | Implementation |
|---|---|
| Dataset & versioning | `data/raw` tracked with DVC (`data/raw.dvc`). `data/processed` and the model checkpoints are DVC pipeline outputs. All are pushed to a DVC remote (local now, S3 in a later phase). |
| Pipeline | `dvc.yaml` has these stages: `validate → preprocess → train@{resnet50, efficientnet_b0} → evaluate@{…} → register`. All hyperparameters live in `params.yaml`. `dvc repro` re-runs only the stages whose code, data or parameters changed. |
| Experiment tracking | MLflow logs parameters, per-epoch training and validation curves, and test metrics: accuracy, QWK, macro-F1, per-class and referable-DR sensitivity/specificity, and macro AUC-ROC. Artifacts include the confusion matrix, training history and the model. Each run is tagged with its git commit and DVC data hash. |
| Model registry | Both models are registered in the MLflow Model Registry as versions of `retinaops-dr-classifier`, with tags for architecture, test scores, data version and commit. The best model by test QWK gets the `production` alias, but only if it beats the current production version. |

## Pipeline stages

| Stage | What it does | Outputs |
|---|---|---|
| `validate` | Checks for corrupt or truncated images, a minimum image size, invalid labels, duplicates and class balance. The pipeline stops if a check fails. | `reports/data_validation.json` |
| `preprocess` | Crops the black border around the fundus, resizes, and makes a stratified 70/15/15 train/val/test split. | `data/processed/`, `reports/split_summary.json` |
| `train@<arch>` | Fine-tunes a pretrained timm model. Uses a class-weighted loss, label smoothing, OneCycle LR, mixed precision, and early stopping on validation QWK. | `models/<arch>/model.pt`, `reports/<arch>/history.csv` |
| `evaluate@<arch>` | Scores the held-out test set and logs the metrics and model to MLflow. | `reports/<arch>/metrics.json`, `confusion_matrix.png` |
| `register` | Registers model versions and promotes the best one. | `reports/registry.json` |

### Handling class imbalance

Grade 0 makes up about half of APTOS, so a model that always predicts "No DR" already scores
49% accuracy. Every DR dataset is skewed this way because it reflects real disease
prevalence. Kaggle's pre-oversampled versions copy images before the train/test split,
which leaks near-duplicates into the test set. So instead, the pipeline:

- measures the imbalance in `validate`
- keeps the test distribution realistic with stratified splits
- trains with inverse-frequency class weights (`train.weighted_sampler: true` switches to oversampling)
- selects and compares models on quadratic weighted kappa and macro-F1, not accuracy

## Models

| Role | Architecture | Notes |
|---|---|---|
| Baseline | ResNet-50 (ImageNet-pretrained) | Carried over from earlier experiments |
| Main | EfficientNet-B0 (ImageNet-pretrained) | Same pipeline, per-model LR and batch size in `params.yaml` |

Training uses PyTorch and timm, which lets us reuse our earlier validated PyTorch training code.

## Quick start

```bash
# 1. Environment (Python 3.12, managed by uv)
uv sync

# 2. Data: pull the versioned dataset from the DVC remote ...
uv run dvc pull
#    ... or download it from Kaggle and track it (needs ~/.kaggle/kaggle.json)
uv run python -m retinaops.data.download && uv run dvc add data/raw

# 3. Run the whole pipeline (CUDA, Apple MPS or CPU)
uv run dvc repro

# 4. Inspect results
uv run dvc metrics show            # test metrics for both models
uv run dvc params diff             # what changed since the last commit
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

To run an experiment, change a value in `params.yaml` (for example `train.epochs` or
`models.efficientnet_b0.lr`) and run `dvc repro` again. Only the affected stages re-run,
and each training gets its own MLflow run.

## Results

These come from a pipeline run on an Apple M1 Max (MPS), 15 epochs per model, scored on the held-out test set (n = 550).

| Model | Accuracy | QWK | Macro-F1 | Macro AUC | Referable DR sens. / spec. | Registry |
|---|---|---|---|---|---|---|
| ResNet-50 (baseline) | 0.751 | **0.862** | 0.602 | **0.916** | 0.789 / 0.957 | v1, `production` |
| EfficientNet-B0 | **0.758** | 0.850 | 0.596 | 0.867 | **0.803** / 0.957 | v2 |

Per-class sensitivity on the test set:

| Model | No DR | Mild | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| ResNet-50 | 0.97 | 0.64 | 0.51 | 0.52 | 0.52 |
| EfficientNet-B0 | 0.97 | 0.63 | 0.58 | 0.34 | 0.52 |

Training took about 9 minutes for ResNet-50 (~35 s/epoch) and about 6 minutes for EfficientNet-B0 (~23 s/epoch).
The two models are close. ResNet-50 has the higher test QWK, so the registry promoted it to
`production`. EfficientNet-B0 catches more referable cases but misses more Severe ones. The
rare grades (Severe and Proliferative) are still the weak spot.

Load the production model:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pytorch.load_model("models:/retinaops-dr-classifier@production")
```

Full numbers are in `reports/<arch>/metrics.json`, confusion matrices in `reports/<arch>/confusion_matrix.png`,
and the promotion decision in `reports/registry.json`.

## Repository layout

```
params.yaml                  all pipeline parameters
dvc.yaml / dvc.lock          pipeline definition and the exact hashes of the last run
data/raw.dvc                 pointer to the versioned raw dataset
src/retinaops/
  data/download.py           Kaggle → data/raw
  data/validate.py           data quality checks
  data/preprocess.py         crop, resize, stratified split
  data/dataset.py            PyTorch Dataset and augmentations
  model.py                   timm model factory and checkpoint I/O
  metrics.py                 QWK, macro-F1, sensitivity/specificity, AUC
  train.py                   training loop with MLflow tracking
  evaluate.py                test-set evaluation, logs the model to MLflow
  register.py                MLflow Model Registry versioning and promotion
tests/                       pytest unit tests (data checks, split, metrics, model I/O)
reports/                     metrics, plots and validation reports (git-tracked)
```

## Roadmap

- **Phase 2:** FastAPI inference service (`/predict`, `/health`, `/model-info`), a Streamlit UI with Grad-CAM, and Docker images for training and serving.
- **Phase 3:** GitHub Actions CI/CD (pytest → build → ECR), deployment to AWS App Runner or ECS Fargate, CloudWatch monitoring, and moving the DVC remote to S3.

*Proof of concept for a university MLOps course. Not a medical device.*
