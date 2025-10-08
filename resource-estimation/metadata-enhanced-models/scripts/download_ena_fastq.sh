#!/usr/bin/env bash

# Exit if any command fails
set -euo pipefail

# --- Usage check ---
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ACCESSION> <OUTPUT_DIR>"
  echo "Example: $0 SRR16350208 ./data"
  exit 1
fi

ACCESSION="$1"
OUTDIR="$2"

# Create output directory if not exists
mkdir -p "$OUTDIR"

# ENA API endpoint
API_URL="https://www.ebi.ac.uk/ena/portal/api/filereport"
FIELDS="run_accession,fastq_ftp,fastq_md5,fastq_bytes"

# Fetch metadata
echo "Fetching file information for ${ACCESSION}..."
response=$(curl -s "${API_URL}?accession=${ACCESSION}&result=read_run&fields=${FIELDS}")

# Extract fastq_ftp field (skip header)
fastq_urls=$(echo "$response" | awk 'NR>1 {print $2}' | tr ';' '\n')

if [[ -z "$fastq_urls" ]]; then
  echo "No FASTQ URLs found for accession ${ACCESSION}."
  exit 1
fi

# Download each FASTQ file
echo "Downloading FASTQ files to ${OUTDIR}..."
while IFS= read -r ftp_path; do
  # Construct full URL
  url="https://${ftp_path}"
  echo "Downloading: ${url}"
  wget -c -P "$OUTDIR" "$url"
done <<< "$fastq_urls"

echo "✅ Download completed for ${ACCESSION}."