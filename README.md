# FAIRyMAGs

**A FAIR Galaxy Metagenome-Assembled Genomes (MAGs) Workflow**

---

## Step-by-step MAGs Generation Guideline

### Input

* Paired short reads (e.g., [Zenodo dataset](https://zenodo.org/records/15089018))

---

### Main Workflows and Tools

1. **Upload your data to Galaxy**

   * [Galaxy data upload guide](https://training.galaxyproject.org/training-material/faqs/galaxy/#data%20upload)
   * *Note:* Only [usegalaxy.eu](https://usegalaxy.eu) currently supports all tools and databases.
   * Optionally, use dedicated fetch tools for published data: [SRA fastq_dump](https://usegalaxy.eu/root?tool_id=toolshed.g2.bx.psu.edu/repos/iuc/sra_tools/fastq_dump/3.1.1+galaxy1)

2. **Group your data**

   * Create a [paired collection](https://training.galaxyproject.org/training-material/faqs/galaxy/collections_build_list_paired.html)

3. **Run quality control (QC) workflow**

   * [QC workflow](https://iwc.galaxyproject.org/workflow/short-read-qc-trimming-main/)

4. **Optional: Remove host contamination**

   * [Host contamination removal workflow](https://iwc.galaxyproject.org/workflow/host-contamination-removal-short-reads-main/)

5. **Optional: Group reads for co/grouped assembly**

   * [Read grouping tool](https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Ffastq_groupmerge%2Ffastq_groupmerge%2F1.0.2%2Bgalaxy0&version=latest)

6. **Run MAGs generation workflow**

   * [MAGs-building workflow](https://iwc.galaxyproject.org/workflow/mags-building-main/)

---

### Potential Downstream Tools or Workflows

#### MAGs Annotation

* [Taxonomy annotation](https://iwc.galaxyproject.org/workflow/mags-taxonomy-annotation-main/)
* [AMR gene detection](https://iwc.galaxyproject.org/workflow/amr_gene_detection-main/)
* [Bacterial genome annotation](https://iwc.galaxyproject.org/workflow/bacterial_genome_annotation-main/)
* [Functional annotation of protein sequences](https://iwc.galaxyproject.org/workflow/functional-annotation-protein-sequences-main)

#### Differential Abundance Analysis

* [MaAsLin 2 Differential Analysis](https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Fmaaslin2%2Fmaaslin2%2F1.18.0%2Bgalaxy0&version=latest)
* [MaAsLin 3 Differential Analysis](https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Fmaaslin3%2Fmaaslin3%2F0.99.16%2Bgalaxy0&version=latest)

---

### Run Analysis Scripts

1. **Install conda**

   * [Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. **Install dependencies**

```bash
conda create -n fairymags_env -c conda-forge -c bioconda --file requirements.txt -y
```

---

### MetasSPADES Resource Estimation

* Follow the step-by-step guide in the [metaspades-resource-estimation README](metaspades-resource-estimation/README.md)

---

### Pipeline Benchmark

* Follow the step-by-step guide in the [pipeline-benchmark README](pipeline-benchmark/README.md)

