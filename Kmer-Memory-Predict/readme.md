# Step-by-step workflow

* Get memory vs ID file from EBI: [mgnify_assemblies_stats](input/mgnify_assemblies_stats.csv)
* Add SRR ID and subset samples notebook: [add_SSR_to_assembly_stats.ipynb](add_SSR_to_assembly_stats.ipynb)
* Upload SRR ID file to Galaxy, Example: https://usegalaxy.eu/u/paulzierep/h/kmer-counting-subset-3-15-3-metaspades-v2
* Run SRR to kmer workflow: TODO
* Download kmers 
* Run kmer_stats_vs_peak_memory.ipynb

# TODO

* Add kmer stats tool to galaxy [x]
* Add unique kmers [x]
* Run with larger kmer count

# Notes

## Get raw reads for EBI Accession

Example using MGnify API
https://www.ebi.ac.uk/metagenomics/api/v1/assemblies/ERZ500991?format=json
Using ENA API: 
https://www.ebi.ac.uk/ena/browser/api/xml/ERZ500991?download=false&includeLinks=false