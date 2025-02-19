# CAMI Binning Benchmark Workflow

## Public Link

* https://usegalaxy.eu/u/santinof/w/fairymags-taxonomic-binning-evaluation

## Contained subworkflow

* https://usegalaxy.eu/u/santinof/w/fairymags-gtdb-tk-subworkflow

## Meta Data Checklist

* [] License  
* [] Creator 

## Test Data


## IWC PR / Link

* [] TODO

## Benchmark Data

* [cami II marine (Pooled)](https://usegalaxy.eu/u/santinof/h/fairymags-taxonomic-binning-evaluation-pooled) (v1.0)
* [cami II marine (not Pooled)](https://usegalaxy.eu/u/santinof/h/fairymags-taxonomic-binning-evaluation-not-pooled) (v1.0)

## TODO

* [x] Compare result of Binette and DAS Tool
* [x] Compare result of both Maxbin2 (had different coverage table inputs)
* [x] Add Maxbin2 to DAS Tools and/or Binette 
* [ ] Copy Workflow and remove taxonomic binning part to have a binning benchmark workflow 
* [ ] Change Binette to use DM when DB was downloaded
* [ ] Change SemiBin2 DB when new DM was pushed (when there is a PR -> currently looking at it)

## Change log

* MaxBin2 was added to Binette and DAS Tool -> maybe even better results now
* Correct some labels in AMBER
* Used auto-update to update all tools to the current version
* Swap the GTDB-subworkflow to the correct workflow

### Version 1.1
* Currently for Binette the CheckM2 DB need to be manually inputted since DM is not working in Galaxy
* Rear error can happen in a GTDB-Tk run (happen only once yet)
