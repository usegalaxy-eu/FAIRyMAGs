import os
import pandas as pd
from Bio import SeqIO
import plotly.graph_objects as go
from pathlib import Path

def gtdbtk_summaries_to_dict(base_dir):

    base_path = Path(base_dir)
    genome_dict = {}

    for summary_file in base_path.rglob("*summary.tsv"):
        try:
            df = pd.read_csv(summary_file, sep='\t')
        except Exception as e:
            print(f"Skipping {summary_file}: {e}")
            continue

        if 'user_genome' not in df.columns or 'classification' not in df.columns:
            print(f"Skipping {summary_file}: missing required columns")
            continue

        def extract_taxonomy(tax_string, rank):
            if pd.isna(tax_string) or tax_string == "":
                return None
            parts = tax_string.split(";")
            rank_dict = {}
            for p in parts:
                if "__" in p:
                    r, name = p.split("__", 1)
                    rank_dict[r] = name
            return rank_dict.get(rank, None)

        for _, row in df.iterrows():
            genome = row['user_genome']
            phylum = extract_taxonomy(row['classification'], 'p')
            clazz = extract_taxonomy(row['classification'], 'c')
            genome_dict[genome] = {'phylum': phylum, 'class': clazz}

    return genome_dict

def collect_bin_assignments(base_dir, classification):

    records = []

    for tool in os.listdir(base_dir):
        tool_path = os.path.join(base_dir, tool)
        if not os.path.isdir(tool_path):
            continue

        for fasta_file in os.listdir(tool_path):
            if not fasta_file.lower().endswith((".fa", ".fasta", ".fna")):
                continue

            if tool == 'dRep':
                phylum = classification[fasta_file.replace('.', '_')]['phylum']
                clazz = classification[fasta_file.replace('.', '_')]['class']
            else:
                phylum = classification[os.path.splitext(fasta_file)[0]]['phylum']
                clazz = classification[os.path.splitext(fasta_file)[0]]['class']

            fasta_path = os.path.join(tool_path, fasta_file)
            for record in SeqIO.parse(fasta_path, "fasta"):
                records.append({
                    "Contig ID": record.id,
                    "Tool": tool,
                    "Bin ID": os.path.splitext(fasta_file)[0],
                    'Phylum': phylum,
                    'Class': clazz
                })

    df = pd.DataFrame(records)
    return df

def plot_sanky(df):

    nodes = pd.concat([df['Tool'], df['Bin ID'], df['Phylum'], df['Class']]).unique().tolist()
    node_indices = {node: i for i, node in enumerate(nodes)}

    links = []

    def add_links(col1, col2):
        grouped = df.groupby([col1, col2]).size().reset_index(name='count')
        for _, row in grouped.iterrows():
            source = node_indices[row[col1]]
            target = node_indices[row[col2]]
            value = row['count']
            links.append({'source': source, 'target': target, 'value': value})

    add_links('Tool', 'Bin ID')
    add_links('Bin ID', 'Phylum')
    add_links('Phylum', 'Class')

    sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes
        ),
        link=dict(
            source=[l['source'] for l in links],
            target=[l['target'] for l in links],
            value=[l['value'] for l in links]
        )
    )])

    sankey.update_layout(title_text="Contig Fate and Taxonomy Sankey", font_size=10)
    sankey.write_html("sankey_plot.html")

if __name__ == '__main__':
    classification = gtdbtk_summaries_to_dict('binner')
    df = collect_bin_assignments('binner', classification)
    print(df.head())
    plot_sanky(df)