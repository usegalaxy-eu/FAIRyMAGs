import os
import pandas as pd
from Bio import SeqIO
import plotly.graph_objects as go


def collect_bin_assignments(base_dir):

    records = []

    for tool in os.listdir(base_dir):
        tool_path = os.path.join(base_dir, tool)
        if not os.path.isdir(tool_path):
            continue

        for fasta_file in os.listdir(tool_path):
            if not fasta_file.lower().endswith((".fa", ".fasta", ".fna")):
                continue

            fasta_path = os.path.join(tool_path, fasta_file)
            for record in SeqIO.parse(fasta_path, "fasta"):
                records.append({
                    "Contig ID": record.id,
                    "Tool": tool,
                    "Bin ID": os.path.splitext(fasta_file)[0]
                })

    df = pd.DataFrame(records)
    return df

def plot_sanky(df):

    df['node'] = df['Tool'] + ':' + df['Bin ID']
    nodes = list(df['node'].unique())

    node_index = {node: n for n, node in enumerate(nodes)}

    links = []
    binners = df['Tool'].unique()

    for i in range(len(binners) - 1):
        left = binners[i]
        right = binners[i+1]
        df_left = df[df['Tool'] == left].set_index('Contig ID')
        df_right = df[df['Tool'] == right].set_index('Contig ID')
        common = df_left.join(df_right, lsuffix='_l', rsuffix='_r')
        flows = common.groupby(['node_l', 'node_r']).size().reset_index(name='count')

        for _, row in flows.iterrows():
            links.append({
                'source': node_index[row['node_l']],
                'target': node_index[row['node_r']],
                'value': row['count']
            })

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=nodes
            ),
            link=dict(
                source=[l['source'] for l in links],
                target=[l['target'] for l in links],
                value=[l['value'] for l in links]
            )
        )])

        fig.write_html("sankey_plot.html")

if __name__ == "__main__":
    df = collect_bin_assignments("binner")
    plot_sanky(df)
    print(df.head())