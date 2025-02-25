# CAMI Binning Benchmark Workflow

## Public Link

* https://usegalaxy.eu/u/santinof/w/fairymags-taxonomic-binning-evaluation-v12

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
* [x] Copy Workflow and remove taxonomic binning part to have a binning benchmark workflow 
* [x] Change Binette to use DM when DB was downloaded
* [ ] Change SemiBin2 DB when new DM was pushed (when there is a PR -> currently looking at it)

## Change log

* MaxBin2 was added to Binette and DAS Tool -> maybe even better results now
* Correct some labels in AMBER
* Used auto-update to update all tools to the current version
* Swap the GTDB-subworkflow to the correct workflow
# 1.1 -> 1.2

* Change Binette DB used to DM since DM is fixed

### Version 1.2
* Rear error can happen in a GTDB-Tk run (happen only once yet)
* Some inputted data got corrupt possible Galaxy problem when having a big dataset? (happen to the pooled dataset, after setting them up new the error which did happen at one step is not happening now)
