Here’s a cleaned-up, nicely formatted version of your notes with consistent headings, bullet points, and a clearer flow:
# Idea

Generate **MAGs** for the **CAMI II plant dataset** using different pipelines:

* **FAIRyMAGs workflow**: [Galaxy history](https://usegalaxy.eu/u/paulzierep/h/drep-benchmark-1-1-1-1-1)
* **MGnify** [x]
* **Magneto** [x]
* **nf-core** TODO

Merge all MAGs with the reference genomes and compute metrics:

* Number of MAGs clustering with the true reference genomes
* Overlap between workflows

**Goal:** Estimate how many high-quality MAGs can be recovered that cluster at the species level with the true genomes.

---

# Workflow

* [FAIRyMAGs benchmark workflow](https://usegalaxy.eu/u/paulzierep/w/drep-fairymags-benchmark)

---

# History

## First Try

* Included only **FAIRyMAGs MAGs** and reference genomes
* [Galaxy history](https://usegalaxy.eu/u/paulzierep/h/drep-benchmark-1-1-1-1-1)

## FAIRyMAGs + MGnify + Magneto + Reference Genomes

* [Galaxy history](https://usegalaxy.eu/u/paulzierep/h/drep-benchmark-v2)

---

# Setup Environment

* Use `conda` or your preferred Python environment to install dependencies

---

# Analysis Script

* `drep-benchmark.ipynb`

