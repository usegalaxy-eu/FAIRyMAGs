# Bee gut microbiome use case

Histories: 
* Input: https://usegalaxy.eu/u/paulzierep/h/bee-gut-microbiome-prjna977416
* Quality control of raw data: https://usegalaxy.eu/u/paulzierep/h/short-read-quality-control-and-trimming
* Host and contaminant read removal: https://usegalaxy.eu/u/paulzierep/h/host-or-contamination-removal-on-short-reads
* MAGs building and quality control: https://usegalaxy.eu/u/paulzierep/h/metagenome-assembled-genomes-mags-generation
* Taxonomy classification: https://usegalaxy.eu/u/paulzierep/h/mags-taxonomy-annotation
* Genome annotation: https://usegalaxy.eu/u/paulzierep/h/mag-genome-annotation-parallel
* Functional annotation: https://usegalaxy.eu/u/paulzierep/h/functional-annotation-of-sequences-1

## ENA submission

The MAGs recovered from this use case were deposited in ENA as third-party
(TPA) annotated MAGs under study
[PRJEB128432](https://www.ebi.ac.uk/ena/browser/view/PRJEB128432)
(alias `honeybee-mags-tpa-20261014`), with the raw reads of the source project
[PRJNA977416](https://www.ebi.ac.uk/ena/browser/view/PRJNA977416) as the underlying data.
Each MAG is linked to the run it was assembled from via its identifier
(`<run_accession>_binette_bin<N>`).

`ena_submission_samplesheet.csv` is the samplesheet used for the submission. It was
prepared for the [nf-core/seqsubmit](https://nf-co.re/seqsubmit) pipeline and is
kept here for provenance. It contains one row per MAG with the assembly and binning
software (with versions), the binning parameters that differ from the defaults of the
FAIRyMAGs workflows, the CheckM2 quality statistics, the coverage, the ENA
environmental metadata and the taxonomic lineage. The `fasta` column points to the
MAG FASTA files as staged on the submission host.
