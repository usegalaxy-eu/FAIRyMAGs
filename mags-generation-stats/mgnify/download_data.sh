# Maximum number of parallel downloads
MAX_PARALLEL=4

# Array of files to download
FILES=(
  "rhimgCAMI2_sample_0_bam.tar.gz"
  "rhimgCAMI2_sample_0_contigs.tar.gz"
  "rhimgCAMI2_sample_0_reads.tar.gz"
  "rhimgCAMI2_sample_1_bam.tar.gz"
  "rhimgCAMI2_sample_1_contigs.tar.gz"
  "rhimgCAMI2_sample_1_reads.tar.gz"
  "rhimgCAMI2_sample_2_bam.tar.gz"
  "rhimgCAMI2_sample_2_contigs.tar.gz"
  "rhimgCAMI2_sample_2_reads.tar.gz"
  "rhimgCAMI2_sample_3_bam.tar.gz"
  "rhimgCAMI2_sample_3_contigs.tar.gz"
  "rhimgCAMI2_sample_3_reads.tar.gz"
  "rhimgCAMI2_sample_4_bam.tar.gz"
  "rhimgCAMI2_sample_4_contigs.tar.gz"
  "rhimgCAMI2_sample_4_reads.tar.gz"
  "rhimgCAMI2_sample_5_bam.tar.gz"
  "rhimgCAMI2_sample_5_contigs.tar.gz"
  "rhimgCAMI2_sample_5_reads.tar.gz"
  "rhimgCAMI2_sample_6_bam.tar.gz"
  "rhimgCAMI2_sample_6_contigs.tar.gz"
  "rhimgCAMI2_sample_6_reads.tar.gz"
  "rhimgCAMI2_sample_7_bam.tar.gz"
  "rhimgCAMI2_sample_7_contigs.tar.gz"
  "rhimgCAMI2_sample_7_reads.tar.gz"
  "rhimgCAMI2_sample_8_bam.tar.gz"
  "rhimgCAMI2_sample_8_contigs.tar.gz"
  "rhimgCAMI2_sample_8_reads.tar.gz"
)

# Function to download a file
download_file() {
  local FILE=$1
  echo "Downloading ${FILE}..."
  if curl -C - -O "${BASE_URL}/${FILE}"; then
    echo "Successfully downloaded ${FILE}"
  else
    echo "Failed to download ${FILE}"
  fi
}

# Download files in parallel with limited concurrency
for FILE in "${FILES[@]}"; do
  # Wait if we have reached max parallel downloads
  while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
    sleep 0.1
  done

  # Start download in background
  download_file "$FILE" &
done

# Wait for all background jobs to complete
wait

echo "Download complete!"