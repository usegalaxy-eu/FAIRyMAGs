# MAGs generation

## Note

This workflow requires as input quality-controlled host-removed paired-end reads. QC can be performed with: TODO
For co-assembly / grouped-assembly, the workflow [Group-Co-assembly](Group-Co-assembly.md) can be used.

## Public Link

* [mags-generation](https://usegalaxy.eu/u/paulzierep/w/mags-generation)

## Meta Data Checklist

* [x] License  
* [x] Creator 

## Test Data

* https://zenodo.org/records/15089018 (~ 3MMB) - works only for the MEGAHIT version
* https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-17---minimal-test-data (65 MB) (v1.7)

## IWC PR / Link

* [x] https://github.com/galaxyproject/iwc/pull/769

## Benchmark Data

* [cami II marine](https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-cami-ii-marine-dataset) (v1.7)
* [nf-core](https://usegalaxy.eu/u/paulzierep/h/mags-individual-workflow-1-7-nf-core) (1.7)

## TODO

* [ ] QC subworkflow 

## Change log

### Version 1.20

The workflow supports assembly using **metaSPADES** and **MEGAHIT**.  
For binning, it utilizes four different tools: **MetaBAT2, MaxBin2, SemiBin, and CONCOCT**. The resulting bins are then refined using **Binette**, the successor of metaWRAP.  

After binning, the resulting MAGs are **dereplicated** across all input samples based on **CheckM2 quality metrics**. The following processing steps are then performed:  

- **Annotation** with Bakta  
- **Taxonomic Assignment** using GTDB-Tk  
- **Quality Control** via QUAST and CheckM/CheckM2  
- **Abundance Estimation** per sample with CoverM  

All results are consolidated into a single **MultiQC report** for easy analysis.  

### Version 1.7

* Flatten collection before dRep to get unique MAGs for all samples
* Run Quast on Bins (no Reference) on after dRep 
* Add tags for output
* MultiQC with GTDB-tk, Quast (Bins), CoverM, CheckM
