# FAIRyMAGs

A FAIR Galaxy Metagenome-Assembled Genomes (MAGs) Workflow

# Step-by-step MAGs generation guidline

## Input 
Paired short reads (e.g.: https://zenodo.org/records/15089018)

## Instructions
* [Upload your data to Galaxy](https://training.galaxyproject.org/training-material/faqs/galaxy/#data%20upload) (currently only usegalaxy.eu supports all the tools / DBs) or use a dedicated data fetch tool for published data like [toolshed.g2.bx.psu.edu/repos/iuc/sra_tools/fastq_dump/3.1.1+galaxy1](https://usegalaxy.eu/root?tool_id=toolshed.g2.bx.psu.edu/repos/iuc/sra_tools/fastq_dump/3.1.1+galaxy1).

* Group your data in a [paired collection](https://training.galaxyproject.org/training-material/faqs/galaxy/collections_build_list_paired.html)

* Run: [QC workflow](https://iwc.galaxyproject.org/workflow/short-read-qc-trimming-main/)

* Optionally: [Group reads for co-/grouped-assembly workflow](https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Ffastq_groupmerge%2Ffastq_groupmerge%2F1.0.2%2Bgalaxy0&version=latest)

* Optionally: [Remove host contamination workflow](https://iwc.galaxyproject.org/workflow/host-contamination-removal-short-reads-main/)

* Run: [MAGs-genertion workflow](https://iwc.galaxyproject.org/workflow/mags-building-main/)

### Potential Downstream Tools or Workflows

* [AMR gene detection](https://iwc.galaxyproject.org/workflow/amr_gene_detection-main/)
* [Bacterial genome annotation](https://iwc.galaxyproject.org/workflow/bacterial_genome_annotation-main/)
* [Functional annotation of protein sequences](https://iwc.galaxyproject.org/workflow/functional-annotation-protein-sequences-main)
* [Differential Analysis with MaAsLin 2](https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Fmaaslin2%2Fmaaslin2%2F1.18.0%2Bgalaxy0&version=latest)

# MAGs Workflows
* Note: Until the workflows are published on IWC the latest version of the workflows as published on usegalaxy.eu should be documented here: [Workflows](Workflows)
* Please Tag related workflows with: `#FAIRyMAGs`
