
# Task 1: Overview
Identify relevant metadata features that can enhance the estimation models


# Potential metadata features with notes:

**Intrinsic to datafiles**

- File size:
    - ML Feature:
        - Log transform
        - MaxMin scaling
        - Raw bytes
    - Note: easy to compute, alrady showed good predictiive power independently
    
- Number of reads:
    - ML Feature:
        - Log transform
        - Normalise by number of reads
        - Raw count
    - Note: easy to compute

- Read length stats (e.g., median, N50, L50):
    - ML Feature:
        - Median read length
        - N50
        - L50
    - Note: A bit more resource intensive, but still easy to compute. Requires external tools

- GC content:
    - ML Feature:
        - Fraction of G and C bases
    - Note: A bit more resource intensive, but still easy to compute.

**Extrinsic to datafiles**
The following could be retrived from metadata associated with the sequencing project or sample. if the ENA accession is known.

    - Sequencing platform (e.g., Illumina, PacBio, Oxford Nanopore)
    - Library preparation method
    - Sample biome (e.g., soil, water, gut)


# Prioritised features
1. File size
2. Number of reads
3. Read length stats (e.g., median, N50, L50)
4. GC content

**Extrinsic features** will have low priority as they may not be consistently available across datasets and may require additional effort to retrieve.
