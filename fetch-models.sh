#!/usr/bin/env bash
# Download the ONNX models used for speaker diarization (~37 MB).
# Only needed if you plan to use --diarize; Whisper models fetch themselves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p models && cd models

BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download
SEG="$BASE/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
# Note: "recongition" is spelled that way in the upstream release tag.
EMB="$BASE/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"

if [[ ! -d sherpa-onnx-pyannote-segmentation-3-0 ]]; then
  echo "Fetching speaker segmentation model..."
  curl -fsSL -o seg.tar.bz2 "$SEG"
  tar xjf seg.tar.bz2
  rm -f seg.tar.bz2
fi

if [[ ! -f campplus_en.onnx ]]; then
  echo "Fetching speaker embedding model..."
  curl -fsSL -o campplus_en.onnx "$EMB"
fi

echo "Models ready:"
find . -name "*.onnx" -printf "  %p (%s bytes)\n"
