#!/usr/bin/env bash
# Use-case plots with mags-visualization (https://github.com/usegalaxy-eu/MAGs-visualization).
#
# For every use case and setup this draws the page-width overview figure and
# the single plots from the campaign tables in data/use-cases/<use case>/<setup>/
# into results/use-cases/<use case>/<setup>/. Each command is the matching
# example of the MAGs-visualization gallery (docs/plots.md), with the use-case
# tables and metadata; the gallery shows what every plot looks like.
#
# A setup is only plotted when the rerun campaign has exported the tables of all
# four of its stages (MAGs, taxonomy, genome and functional annotation), so the
# plots never mix the new MAGs with taxonomy or annotation of an older run.
#
#   bin/make_use_case_plots.sh --check              # what is ready, draw nothing
#   bin/make_use_case_plots.sh                      # every ready setup
#   bin/make_use_case_plots.sh bee-gut-microbiome/multi
#   bin/make_use_case_plots.sh --force aeromicrobiome/megahit   # skip the campaign check
#
# Environment:
#   MAGS_VIS   mags-visualization executable (default: the mags-vis-dev conda env)
#   CAMPAIGN   campaign folder with table-exports/ (default: data/campaigns/20261122)
#   OUT_ROOT   where the plots go (default: results/use-cases; set it for test renders)
#   DATA_ROOT  where the tables are (default: data/use-cases; e.g. an archived run, with --force)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAGS_VIS=${MAGS_VIS:-$HOME/miniforge3/envs/mags-vis-dev/bin/mags-visualization}
CAMPAIGN=${CAMPAIGN:-$REPO/data/campaigns/20261122}
OUT_ROOT=${OUT_ROOT:-$REPO/results/use-cases}
DATA_ROOT=${DATA_ROOT:-$REPO/data/use-cases}
PYTHON=${PYTHON:-python3}
FORMATS=(png svg)
STAGES=(mags taxonomy genome_annotation functional_annotation)
TABLES=(gtdb.tsv coverm.tsv checkm2.tsv drep.tsv kegg_pathway_completeness.tsv quast.tsv bakta.tsv)

# use case/setup, in campaign order
RUNS=(
  macroalgal-epiphytic-microbiome/single
  bee-gut-microbiome/single
  bee-gut-microbiome/multi
  termite-head-microbiome/grouped
  termite-head-microbiome/individual
  aeromicrobiome/megahit
  aeromicrobiome/metaspades
)

check=false
force=false
selected=()
for arg in "$@"; do
  case $arg in
    --check) check=true ;;
    --force) force=true ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "Unknown option $arg" >&2; exit 2 ;;
    *) selected+=("${arg%/}") ;;
  esac
done
if [ ${#selected[@]} -eq 0 ]; then selected=("${RUNS[@]}"); fi
for run in "${selected[@]}"; do
  [[ " ${RUNS[*]} " == *" $run "* ]] || { echo "Unknown run $run (known: ${RUNS[*]})" >&2; exit 2; }
done

# Metadata per use case: the file, the columns shown in the abundance panel and
# extra overview-figure options. Macroalgal has 3 samples and no metadata.
metadata_args() {  # $1 use case, $2 data dir, $3 tmp dir
  case $1 in
    bee-gut-microbiome)
      # The sample table has no short label; derive the one of the MAGs-visualization
      # test data: condition + replicate number (NT3 = NT, Replicate3).
      local meta=$3/bee_metadata.tsv
      awk -F'\t' -v OFS='\t' 'NR == 1 {
          for (i = 1; i <= NF; i++) col[$i] = i
          if (!("Condition" in col) || !("Replicate" in col)) { print "bee metadata: no Condition/Replicate column" > "/dev/stderr"; exit 1 }
          print $0, "Label"; next }
        { rep = $col["Replicate"]; sub(/.*[^0-9]/, "", rep); print $0, $col["Condition"] rep }' \
        "$REPO/data/use-cases/bee-gut-microbiome/single/metadata.tsv" > "$meta"
      META=(--metadata "$meta" --meta_cols "Infection by Nosema ceranae" "Chronic exposure to neonicotinoid" "Treatment with probiotic")
      OVERVIEW_EXTRA=(--meta_labels "N. ceranae" Neonicotinoid Probiotic
                      --sample_label_col Label --sample_group_col Condition --group_order C P T TP N NP NT) ;;
    termite-head-microbiome)
      META=(--metadata "$REPO/data/use-cases/termite-head-microbiome/metadata.tsv" --meta_cols Species Casts Colony)
      OVERVIEW_EXTRA=() ;;
    aeromicrobiome)
      META=(--metadata "$REPO/data/use-cases/aeromicrobiome/metadata.tsv"
            --meta_cols "Geographic origin of the air mass" Condition Season)
      OVERVIEW_EXTRA=() ;;
    *) META=(); OVERVIEW_EXTRA=() ;;
  esac
}

# 0 when all four stages of the setup are exported (table-exports/<stage>.json).
campaign_ready() {  # $1 use case, $2 setup
  "$PYTHON" - "$CAMPAIGN/table-exports" "$1" "$2" "${STAGES[@]}" <<'EOF'
import json, sys
from pathlib import Path
folder, case, setup, stages = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
missing = []
for stage in stages:
    path = folder / f"{case}.{stage}_{setup}.json"
    if not path.exists() or not json.loads(path.read_text()).get("collection_complete"):
        missing.append(stage)
if missing:
    print("not exported yet: " + ", ".join(missing))
    sys.exit(1)
EOF
}

[ -x "$MAGS_VIS" ] || { echo "mags-visualization not found at $MAGS_VIS (set MAGS_VIS)" >&2; exit 1; }
echo "mags-visualization $("$MAGS_VIS" --version 2>&1 | tail -1) ($MAGS_VIS)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

plotted=() skipped=() failed=()
for run in "${selected[@]}"; do
  case=${run%%/*} setup=${run#*/}
  d=$DATA_ROOT/$run
  out=$OUT_ROOT/$run
  missing=()
  for t in "${TABLES[@]}"; do [ -s "$d/$t" ] || missing+=("$t"); done
  if [ ${#missing[@]} -gt 0 ]; then
    skipped+=("$run: missing ${missing[*]}"); continue
  fi
  if ! $force && ! reason=$(campaign_ready "$case" "$setup"); then
    skipped+=("$run: $reason"); continue
  fi
  if $check; then plotted+=("$run (ready)"); continue; fi

  echo "== $run -> ${out#$REPO/}"
  mkdir -p "$out"
  metadata_args "$case" "$d" "$tmp"
  log=$out/mags-visualization.log
  : > "$log"
  step() {  # run one subcommand, keep going if it fails
    if "$MAGS_VIS" "$@" -o "$out" >> "$log" 2>&1; then echo "   ok   $1"; else echo "   FAIL $1 (see ${log#$REPO/})"; return 1; fi
  }
  ok=true
  step overview-figure --gtdb "$d/gtdb.tsv" --coverm "$d/coverm.tsv" --checkm2 "$d/checkm2.tsv" \
       --drep "$d/drep.tsv" --pathways "$d/kegg_pathway_completeness.tsv" \
       "${META[@]}" "${OVERVIEW_EXTRA[@]}" --flow_drop_unclassified --format "${FORMATS[@]}" || ok=false
  # Abundance, quality and function in both styles: classic and as the overview panel.
  step sample-heatmap --coverm "$d/coverm.tsv" --gtdb "$d/gtdb.tsv" "${META[@]}" --tax_level phylum || ok=false
  step sample-heatmap --style overview --coverm "$d/coverm.tsv" --gtdb "$d/gtdb.tsv" \
       "${META[@]}" "${OVERVIEW_EXTRA[@]}" --width_cm 12 || ok=false
  step comp-conta --tools checkm2 --checkm2 "$d/checkm2.tsv" --mode quality || ok=false
  step comp-conta --style overview --tools checkm2 --checkm2 "$d/checkm2.tsv" --mq_contamination 10 || ok=false
  # Static: the overview-style flow diagram. The classic Sankey stays
  # interactive HTML (its static export needs kaleido).
  step taxa-sankey --style overview --gtdb "$d/gtdb.tsv" --flow_drop_unclassified --format png || ok=false
  step taxa-sankey --gtdb "$d/gtdb.tsv" --order_links --font_size 14 --format html || ok=false
  step drep-cluster-annot --drep "$d/drep.tsv" --gtdb "$d/gtdb.tsv" --checkm2 "$d/checkm2.tsv" \
       --quast "$d/quast.tsv" --bakta "$d/bakta.tsv" --top_n 30 || ok=false
  step drep-cluster-func --drep "$d/drep.tsv" --gtdb "$d/gtdb.tsv" \
       --pathways "$d/kegg_pathway_completeness.tsv" --top_n 30 || ok=false
  step drep-cluster-func --style overview --drep "$d/drep.tsv" --gtdb "$d/gtdb.tsv" \
       --pathways "$d/kegg_pathway_completeness.tsv" --top_n 20 || ok=false
  step pathway-module-heatmap --drep "$d/drep.tsv" --gtdb "$d/gtdb.tsv" \
       --pathways "$d/kegg_pathway_completeness.tsv" || ok=false
  if $ok; then plotted+=("$run"); else failed+=("$run"); fi
done

echo
for r in "${plotted[@]}"; do echo "plotted  $r"; done
for r in "${skipped[@]}"; do echo "skipped  $r"; done
for r in "${failed[@]}"; do echo "failed   $r"; done
[ ${#failed[@]} -eq 0 ]
