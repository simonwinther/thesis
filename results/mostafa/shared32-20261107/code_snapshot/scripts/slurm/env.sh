# Shared settings for the WFM-InformedSource Slurm scripts (sourced, not run).
CONDA_SH=${CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-wmf}
TRAIN_DIR=${TRAIN_DIR:-$HOME/datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData}
VAL50_DIR=${VAL50_DIR:-$HOME/datasets/subsets/val50/data}
VALFULL_DIR=${VALFULL_DIR:-$HOME/datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData}

# Set VENV (e.g. on UCloud) to use a Python venv instead of conda.
activate_env() {
  set +u
  if [[ -n ${VENV:-} ]]; then
    source "$VENV/bin/activate"
  else
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
  fi
  set -u
}
