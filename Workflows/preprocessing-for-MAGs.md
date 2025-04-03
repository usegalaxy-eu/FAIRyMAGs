# MAGs QC

## Note

This workflow requires raw paried end reads and performes QC using fastp.

## Public Link

* [preprocessing-for-mags](https://usegalaxy.eu/u/paulzierep/w/preprocessing-for-mags)

## Meta Data Checklist

* [x] License  
* [x] Creator 

## Test Data

* https://zenodo.org/records/15089018 (~ 3MB)

## IWC PR / Link

* [ ] TODO

## Benchmark Data

* [cami II marine](https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-cami-ii-marine-dataset) (v1.7)
* [nf-core](https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-1-7-nf-core) (1.7)

## Change log

### Version 1.19

Uses fastp for QC and falco for QC stats as well as multiQC to show quality before and after QC.
