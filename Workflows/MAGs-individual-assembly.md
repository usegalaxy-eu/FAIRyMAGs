# MAGs individual assembly workflow

## Public Link

* https://usegalaxy.eu/u/paulzierep/w/mags-individual-workflow

## Meta Data Checklist

* [x] License  
* [x] Creator 

## Test Data

* https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-17---minimal-test-data (65 MB) (v1.7)

## IWC PR / Link

* [] TODO
amrfinderplus
## Benchmark Data

* [cami II marine](https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-cami-ii-marine-dataset) (v1.7)
* [nf-core](https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-1-7-nf-core) (1.7)

## TODO

* [ ] QC subworkflow 

## Change log

### Version 1.7

* Flatten collection before dRep to get unique MAGs for all samples
* Run Quast on Bins (no Reference) on after dRep 
* Add tags for output
* MultiQC with GTDB-tk, Quast (Bins), CoverM, CheckM