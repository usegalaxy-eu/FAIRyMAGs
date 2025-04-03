# FAIRyMAGs
FAIRyMAGs - BFSP Project Management

# Step-by-step MAGs generation
## Input 
Paired short reads (e.g.: https://zenodo.org/records/15089018)

## Instructions
* [Upload your data to Galaxy](https://training.galaxyproject.org/training-material/faqs/galaxy/#data%20upload) (currently only usegalaxy.eu supports all the tools / DBs) or use a dedicated data fetch tool for published data like [toolshed.g2.bx.psu.edu/repos/iuc/sra_tools/fastq_dump/3.1.1+galaxy1](https://usegalaxy.eu/root?tool_id=toolshed.g2.bx.psu.edu/repos/iuc/sra_tools/fastq_dump/3.1.1+galaxy1).
* Group your data in a [paired collection](https://training.galaxyproject.org/training-material/faqs/galaxy/collections_build_list_paired.html)
* Perform: [QC workflow](Workflows/QC-for-MAGs.md)
* Optionally: [Remove host contamination workflow](Workflows/MAGs-host-removal)
* Optionally: [Group reads for co-/grouped-assembly workflow](Workflows/Group-Co-assembly.md)
* Perform: [MAGs-genertion workflow](Workflows/MAGs-generation.md)

# MAGs Workflows
* Note: Until the workflows are published on IWC the latest version of the workflows as published on usegalaxy.eu should be documented here: [Workflows](Workflows)
* Please Tag related workflows with: `#FAIRyMAGs`
