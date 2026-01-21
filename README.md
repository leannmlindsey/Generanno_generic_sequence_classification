<p align="center">
  <picture>
    <img alt="Gener" src="figures/logo.png" width=50%>
  </picture>
</p>

<h1 align="center">GENERanno Generic Sequence Classification</h1>

> **Note:** This is a fork of the [original GENERanno repository](https://github.com/GenerTeam/GENERanno) with added support for **generic CSV-based binary/multiclass classification** tasks. This allows you to fine-tune GENERanno on your own DNA sequence classification datasets.

---

## Fine-tuning on Custom CSV Datasets

This fork adds the ability to fine-tune on any classification task using simple CSV files.

### 1. Prepare Your Data

Create a directory containing three CSV files with `sequence` and `label` columns:

```
my_dataset/
├── train.csv
├── dev.csv    # (or val.csv)
└── test.csv
```

Each CSV should have this format:
```csv
sequence,label
ACGTACGTACGT...,0
TGCATGCATGCA...,1
GGCCAATTGGCC...,0
```

- `sequence`: DNA sequence (A, C, G, T, N characters)
- `label`: Integer class label (0, 1 for binary; 0, 1, 2, ... for multiclass)

### 2. Run Fine-tuning

```bash
python -m src.tasks.downstream.sequence_understanding \
    --csv_dir="/path/to/my_dataset" \
    --model_name="GenerTeam/GENERanno-prokaryote-0.5b-base" \
    --output_dir="./results/my_task" \
    --batch_size=16 \
    --max_length=8192 \
    --learning_rate=1e-5 \
    --d_output=2 \
    --seed=42 \
    --main_metrics="mcc"
```

### 3. Random Initialization Baseline

To measure the contribution of pretraining, you can train with randomly initialized weights (same architecture, no pretrained weights). This provides a baseline to compare against the pretrained model.

```bash
python -m src.tasks.downstream.sequence_understanding \
    --csv_dir="/path/to/my_dataset" \
    --model_name="GenerTeam/GENERanno-prokaryote-0.5b-base" \
    --output_dir="./results/my_task_random" \
    --random_init \
    --seed=42
```

When `--random_init` is used:
- The model architecture is loaded from the model config
- Pretrained weights are **not** loaded
- All weights are randomly initialized

**Comparing Results:**

Run two experiments with the same seed and hyperparameters:
1. Pretrained: without `--random_init` (default)
2. Random: with `--random_init`

The difference in metrics shows the "embedding power" gained from pretraining.

### 4. SLURM Scripts (for HPC)

SLURM scripts are provided in `slurm_scripts/` for running on HPC clusters (configured for NIH Biowulf):

**Fine-tuning:**
1. Edit `wrapper_run_generanno_csv.sh` with your paths
2. Submit: `bash wrapper_run_generanno_csv.sh`
3. For interactive testing: `bash run_generanno_csv_interactive.sh`

**Embedding Analysis:**
1. Edit `wrapper_run_embedding_analysis.sh` with your paths
2. Submit: `bash wrapper_run_embedding_analysis.sh`
3. For interactive testing: `bash run_embedding_analysis_interactive.sh`

**Batch Inference:**
1. Create a text file with input CSV paths (one per line)
2. Edit `wrapper_run_batch_inference.sh` with your paths
3. Submit: `bash wrapper_run_batch_inference.sh`
4. For interactive testing: `bash run_batch_inference_interactive.sh`

### 5. Embedding Analysis

Extract embeddings and evaluate their quality with linear probes, silhouette scores, PCA visualization, and a 3-layer NN:

```bash
python -m src.tasks.downstream.embedding_analysis \
    --csv_dir="/path/to/csv/data" \
    --model_path="GenerTeam/GENERanno-prokaryote-0.5b-base" \
    --output_dir="./results/embedding_analysis" \
    --pooling="mean" \
    --include_random_baseline  # Optional: compare with random init
```

**Outputs:**
- `embeddings_pretrained.npz`: Extracted embeddings for train/val/test sets
- `pca_visualization_pretrained.png`: PCA plot showing class separation
- `test_predictions_pretrained.csv`: Test predictions with probabilities
- `three_layer_nn_pretrained.pt`: Trained 3-layer NN model
- `embedding_analysis_results.json`: All metrics including embedding power

**Caching:** Embeddings are cached in `.npz` files. Delete them to re-extract with different settings.

### 6. Inference

Run inference on a CSV file to get predictions with probabilities for threshold analysis:

```bash
python -m src.tasks.downstream.inference \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/finetuned/model" \
    --output_csv="/path/to/predictions.csv" \
    --threshold=0.5 \
    --save_metrics
```

**Output CSV columns:** `sequence`, `label`, `prob_0`, `prob_1`, `pred_label`

**Batch Inference:** For processing multiple files, create a text file listing input CSVs (one per line) and use the batch inference scripts:

```bash
# Create input list
echo "/path/to/test1.csv" > input_files.txt
echo "/path/to/test2.csv" >> input_files.txt

# Edit wrapper_run_batch_inference.sh with paths, then run:
bash slurm_scripts/wrapper_run_batch_inference.sh
```

### 7. Test Results

After training, comprehensive test metrics are saved to `test_results.json`:
- `eval_accuracy`, `eval_precision`, `eval_recall`, `eval_f1`
- `eval_mcc`: Matthews Correlation Coefficient (computed on full test set)
- `eval_sensitivity`, `eval_specificity`, `eval_auc`

---

## Original GENERanno README

The remainder of this README is from the [original GENERanno repository](https://github.com/GenerTeam/GENERanno).

---

## 📰 News

* 📑 **[2025-06-05]** Our paper is now available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.04.656517v1)!
* 🤗 **[2025-05-10]** Our expert model for metagenomic annotation `GENERanno-prokaryote-0.5b-cds-annotator` is now available on [HuggingFace](https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-cds-annotator)!
* 🤗 **[2025-02-11]** Our models `GENERanno-prokaryote-0.5b-base`,
  `GENERanno-eukaryote-0.5b-base` are now available on [HuggingFace](https://huggingface.co/GenerTeam/)!

## 🔭 Overview

![overview](figures/model_overview.png)

In this repository, we present GENERanno, a genomic foundation model featuring a context length of 8k base pairs and
500M parameters, trained on an expansive dataset comprising 386 billion base pairs of eukaryotic DNA. Our evaluations
demonstrate that the GENERanno achieves comparable performance
with [GENERator](https://huggingface.co/GenerTeam/GENERator-eukaryote-1.2b-base) in benchmark evaluations,
including [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks/tree/main), [NT tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised),
and our newly proposed [Gener tasks](https://huggingface.co/GenerTeam), making them the top genomic foundation models in
the field (2025-02).

Beyond benchmark performance, the GENERanno model is meticulously designed with its specialization in gene annotation.
The model efficiently and accurately identifies gene locations, predicts gene function, and annotates gene structure,
highlighting its potential to revolutionize genomic research by significantly enhancing the precision and efficiency of
gene annotation processes.

Please note that the GENERanno is currently in the developmental phase. We are actively refining the model and will
release more technical details soon. Stay tuned for updates!

In this repository, you will find the following model checkpoints:

| Model Name                       | Parameters | Data |     Category     |                    Status                     |
|----------------------------------|:----------:|:----:|:----------------:|:---------------------------------------------:|
| `GENERanno-eukaryote-0.5b-base`  |    0.5B    | 386B |    Eukaryote     | [Available](https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-eukaryote-base) |
| `GENERanno-prokaryote-0.5b-base` |    0.5B    | 715B |    Prokaryote    | [Available](https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-prokaryote-base) |
| `GENERanno-prokaryote-0.5b-cds-annotator` |     0.5B     | 890B |    Prokaryote    |[Available](https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-cds-annotator)|

## 📈 Benchmark Performance

### Coding DNA Sequence (CDS) Annotation — `GENERanno-prokaryote-0.5b-cds-annotator-preview`
![bacteria_annotation_summary](figures/bacteria_annotation_summary.png)
The detailed annotation results are provived [here](https://huggingface.co/datasets/GenerTeam/cds-annotation).

### Sequence Understanding (Classification/Regression) — `GENERanno-prokaryote-0.5b-base`
![benchmarks](figures/prokaryotic_gener_tasks.png)

### Sequence Understanding (Classification/Regression) — `GENERanno-eukaryote-0.5b-base`
![benchmarks](figures/eukaryotic_benchmarks.png)

## 🎯 Quick Start

### Dependencies

* Clone this repo, cd into it

```shell
git clone https://github.com/GenerTeam/GENERanno.git
cd GENERanno
```

* Install requirements with Python 3.10

```shell
pip install -r requirements.txt
```

> If your network cannot access huggingface.co normally, we recommend using the following mirror:
> ```shell
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### Downstream

#### Coding DNA Sequence (CDS) Annotation

To run the coding sequence annotation task on
our [cds annotation dataset](https://huggingface.co/datasets/GenerTeam/cds-annotation/), you can use the
following command:

```shell
# Using all available GPUs (default)
python src/tasks/downstream/cds_annotation.py

# Using specific number of GPUs
python src/tasks/downstream/cds_annotation.py --gpu_count ${NUM_GPUS}

# BF16 for faster inference (recommended if supported)
python src/tasks/downstream/cds_annotation.py --bf16
```
Note: BF16 provides faster inference with minimal accuracy impact on supported hardware.

#### Sequence Understanding (Classification/Regression)

To run the sequence understanding task
on [Gener Tasks](https://huggingface.co/datasets/GenerTeam/gener-tasks), [Prokaryotic Gener Tasks](https://huggingface.co/datasets/GenerTeam/prokaryotic-gener-tasks), [NT Tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised), [Genomic Benchmarks](https://huggingface.co/katarinagresova), [DeepSTARR Enhancer](https://huggingface.co/datasets/GenerTeam/DeepSTARR-enhancer-activity),
you can use the following arguments:

* Gener Tasks / Prokaryotic Gener Tasks
    * `--dataset_name GenerTeam/gener-tasks` or `--dataset_name GenerTeam/prokaryotic-gener-tasks`
    * `--subset_name gene_classification` or `--subset_name taxonomic_classification` or ...
* NT Tasks
    * `--dataset_name InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`
    * `--subset_name H2AFZ` or `--subset_name H3K27ac` or ...
* Genomic Benchmarks
    * `--dataset_name katarinagresova/Genomic_Benchmarks_demo_human_or_worm` or
      `--dataset_name katarinagresova/Genomic_Benchmarks_human_ocr_ensembl` or ...
* DeepSTARR Enhancer Activity
    * `--dataset_name GenerTeam/DeepSTARR-enhancer-activity`
    * `--problem_type regression`

on following command:

```shell
# Using single GPU
python src/tasks/downstream/sequence_understanding.py \
    --model_name GenerTeam/GENERator-eukaryote-1.2b-base \
    --dataset_name ${DATASET_NAME} \
    --subset_name ${SUBSET_NAME} \
    --batch_size ${BATCH_SIZE} \
    --problem_type ${PROBLEM_TYPE} \
    --main_metrics ${MAIN_METRICS}

# Using multiple GPUs on single node (DDP)
torchrun --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    src/tasks/downstream/sequence_understanding.py

# Using multiple GPUs on multiple nodes (DDP)
torchrun --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/tasks/downstream/sequence_understanding.py

# Using DeepSpeed or Full Sharded Data Parallel (FSDP)
torchrun --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/tasks/downstream/sequence_understanding.py \
    --distributed_type deepspeed # or fsdp
```

## 📚 Datasets

* [Eukaryotic Gener Tasks](https://huggingface.co/datasets/GenerTeam/gener-tasks)
* [Prokaryotic Gener Tasks](https://huggingface.co/datasets/GenerTeam/prokaryotic-gener-tasks)
* [CDS Annotation](https://huggingface.co/datasets/GenerTeam/cds-annotation)

## 📜 Citation

```
@article{li2025generanno,
	author = {Li, Qiuyi and Wu, Wei and Zhu, Yiheng and Feng, Fuli and Ye, Jieping and Wang, Zheng},
	title = {GENERanno: A Genomic Foundation Model for Metagenomic Annotation},
	elocation-id = {2025.06.04.656517},
	year = {2025},
	doi = {10.1101/2025.06.04.656517},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/06/05/2025.06.04.656517},
	journal = {bioRxiv}
}
```
