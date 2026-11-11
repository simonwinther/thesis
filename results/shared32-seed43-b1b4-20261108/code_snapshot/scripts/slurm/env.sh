# Shared settings for the WFM-InformedSource Slurm scripts (sourced, not run).
if [[ -z ${CONDA_SH:-} ]]; then
  if [[ -f /opt/anaconda3/etc/profile.d/conda.sh ]]; then
    CONDA_SH=/opt/anaconda3/etc/profile.d/conda.sh
  else
    CONDA_SH=$HOME/miniforge3/etc/profile.d/conda.sh
  fi
fi
CONDA_ENV=${CONDA_ENV:-wmf}
TRAIN_DIR=${TRAIN_DIR:-$HOME/datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData}
VAL50_DIR=${VAL50_DIR:-$HOME/datasets/subsets/val50/data}
VALFULL_DIR=${VALFULL_DIR:-$HOME/datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData}
if [[ ! -d $VALFULL_DIR && -d $HOME/datasets/raW/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData ]]; then
  VALFULL_DIR=$HOME/datasets/raW/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData
fi

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
