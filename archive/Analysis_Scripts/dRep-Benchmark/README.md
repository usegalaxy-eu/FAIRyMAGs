# Idea

Generate MAGs for the CAMI II plant dataset using differen pipelines.

- [x] FAIRyMAGs workflow: https://usegalaxy.eu/u/paulzierep/h/drep-benchmark-1-1-1-1-1
- [x] MGnify
- [x] Magneto
- [x] nf-core TODO

Merge all MAGs together with the reference Genomes. Compute metrics:

* How many MAGs cluster with the true ref. Genomes
* Overlap between workflows

The benchmark estimates how many high quality mags can be recovered that cluster on the species level with the true genomes 

# Workflow

* https://usegalaxy.eu/u/paulzierep/w/drep-fairymags-benchmark

# History

## First Try

Includes only FAIRyMAGs MAGs and ref genomes
* https://usegalaxy.eu/u/paulzierep/h/drep-benchmark-1-1-1-1-1

## FAIRyMAGs + MGnify + Magneto + Ref

* https://usegalaxy.eu/u/paulzierep/h/drep-benchmark-v2

# Set up env

# Anylsis script

drep-benchmark.ipynb