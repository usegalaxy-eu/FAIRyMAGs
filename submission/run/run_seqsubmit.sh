#!/usr/bin/env bash
#
# Run nf-core/seqsubmit for a FAIRyMAGs MAG samplesheet.
#
#   usage: run_seqsubmit.sh <validate|submit> <seqsubmit-checkout> <workdir>
#
#     validate : register the MAG samples/assemblies in ENA and run
#                ena-webin-cli in validate mode (no sequence is uploaded).
#                Always run this first.
#     submit   : upload the FASTA files and obtain the accessions.
#                Resumes the validate run, so nothing is recomputed.
#
# <workdir> must contain samplesheet.csv and the MAG files it refers to
# (e.g. mags/<id>.fasta.gz).
#
# Credentials are read by the pipeline from the Nextflow secrets store, which
# lives in $NXF_HOME.  Set them once:
#
#     nextflow secrets set ENA_WEBIN "Webin-XXXXX"
#     nextflow secrets set ENA_WEBIN_PASSWORD "********"
#
# Credentials never appear on the command line or in a file.
#
set -euo pipefail

MODE="${1:?usage: run_seqsubmit.sh <validate|submit> <seqsubmit-checkout> <workdir>}"
PIPELINE="${2:?path to a nf-core/seqsubmit checkout}"
WORKDIR="${3:?work directory (holds samplesheet.csv)}"

case "$MODE" in
    validate) PARAMS="params.validate.json" ;;
    submit)   PARAMS="params.submit.json" ;;
    *) echo "mode must be 'validate' or 'submit'" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Nextflow needs a JDK; point these at your installation if it is not on PATH.
# export JAVA_HOME=/opt/jdk21
# export PATH="$JAVA_HOME/bin:$PATH"
export NXF_ANSI_LOG=false
: "${NXF_HOME:?set NXF_HOME to a writable dir (the secrets store lives there)}"

echo "mode      : $MODE"
echo "pipeline  : $PIPELINE"
echo "workdir   : $WORKDIR"
echo "params    : $PARAMS"
echo "NXF_HOME  : $NXF_HOME"

# Fail early with a clear message instead of a confusing mid-run error.
nextflow secrets list 2>/dev/null | grep -q ENA_WEBIN || {
    echo "ERROR: Nextflow secret ENA_WEBIN is not set (see the header of this script)" >&2
    exit 1
}
[[ -s "$WORKDIR/samplesheet.csv" ]] || { echo "ERROR: $WORKDIR/samplesheet.csv missing" >&2; exit 1; }

# Validate the samplesheet against the pipeline schema before submitting
# anything to ENA (--help also runs the schema validation).
nextflow run "$PIPELINE" \
    -profile docker \
    -params-file "$SCRIPT_DIR/$PARAMS" \
    -c "$SCRIPT_DIR/limits.config" \
    -name "seqsubmit_$(basename "$WORKDIR")_$MODE" \
    -resume \
    -w "$WORKDIR/work" \
    --input "$WORKDIR/samplesheet.csv" \
    --outdir "$WORKDIR/out" \
    2>&1 | tee "$WORKDIR/run_$MODE.log"

echo "nextflow exit: ${PIPESTATUS[0]}"
echo "results in   : $WORKDIR/out"
