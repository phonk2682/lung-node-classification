#!/usr/bin/env bash
# download_weights.sh — Download pre-trained model checkpoints from Google Drive.
#
# Usage:
#   bash download_weights.sh
set -euo pipefail

GDRIVE_URL="https://drive.google.com/drive/folders/1LyVA8gn6EF71iCeVbYkefPp5J1MYxpIR"
DEST="./weights"

echo "Installing gdown..."
pip install -q gdown

echo "Downloading weights to $DEST/ ..."
gdown --folder "$GDRIVE_URL" --output "$DEST" --remaining-ok

echo ""
echo "Done. Weights saved to: $DEST/"
ls "$DEST"
