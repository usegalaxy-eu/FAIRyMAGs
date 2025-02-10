# CAMI Binning Benchmark Workflow

## Public Link

* https://usegalaxy.eu/u/santinof/w/copy-of-mags-taxonomic-binning-evaluation

## Subworkflows

* https://usegalaxy.eu/u/santinof/w/fairymags-gtdb-tk-subworkflow
* https://usegalaxy.eu/u/santinof/w/gtdb2ncbi-taxid-sub-workflow

## Meta Data Checklist

* [] License  
* [] Creator 

## Test Data


## IWC PR / Link

* [] TODO

## Benchmark Data

* [cami II marine (Pooled)](https://usegalaxy.eu/u/santinof/h/copy-of-mags-taxonomic-binning-evaluation-pooled-fairymag) (v1.0)
* [cami II marine (not Pooled)](https://usegalaxy.eu/u/santinof/h/copy-of-mags-taxonomic-binning-evaluation-not-pooled-fairymag-1) (v1.0)

## TODO

* [] Compare result of Binette and DAS Tool
* [] Compare result of both Maxbin2 (had different coverage table inputs)
* [] Add Maxbin2 to DAS Tools and/or Binette 
* [] Copy Workflow and remove taxonomic binning part to have a binning benchmark workflow 

## Change log

### Version 1.0
* Currently for Binette the CheckM2 DB need to be manually inputed since DM is not working in Galaxy
* Rear error can happen in a GTDB-Tk run (happen only once yet)
