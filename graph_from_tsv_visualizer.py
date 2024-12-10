
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import utils.letter_encoding as letter_encoding
import csv
from adjustText import adjust_text

# import pygraphviz

rows_per_example = 21


def process_df(df: pd.DataFrame, char_to_value_map, max_dur = 1_000_000):

    avg_duration_before = {}
    avg_duration_after = {}
    key_pairs = ([], [])

    key_pairs_to_durs = {}

    for i in range(1, len(df)):
        key_this = df.iloc[i]["key"]
        dur_this = df.iloc[i]['duration']
        key_before = df.iloc[i-1]["key"]
        dur_before = df.iloc[i-1]['duration']
        if dur_this > max_dur:
            dur_this = max_dur

        if dur_before > max_dur:
            dur_before = max_dur

        pair = (key_before, key_this)

        if pair in key_pairs_to_durs:
            key_pairs_to_durs[pair].append(dur_this)
        else:
            key_pairs_to_durs[pair] = [dur_this]
        
        key_pairs[0].append(key_before)
        key_pairs[1].append(key_this)

        # Insert keys
        if not key_this in avg_duration_before:
            avg_duration_before[key_this] = (0, 0)
        if not key_this in avg_duration_after:
            avg_duration_after[key_this] = (0, 0)
        if not key_before in avg_duration_before:
            avg_duration_before[key_before] = (0, 0)
        if not key_before in avg_duration_after:
            avg_duration_after[key_before] = (0, 0)

        s, c = avg_duration_before[key_this]
        avg_duration_before[key_this] = (s + dur_this, c + 1)
        s, c = avg_duration_after[key_before]
        avg_duration_after[key_before] = (s + dur_before, c + 1)

    avg_duration_before = { k:(t[0]/t[1]) if t[1] != 0 else 0 for (k,t) in avg_duration_before.items() }
    avg_duration_after =  { k:(t[0]/t[1]) if t[1] != 0 else 0 for (k,t) in avg_duration_after.items()  }

    df = df[["key", "accel_x","accel_y","accel_z"]].groupby("key").mean().reset_index()
    df["duration_before"] = df["key"].apply(lambda x: avg_duration_before.get(x))
    df["duration_after"] = df["key"].apply(lambda x: avg_duration_after.get(x))

    # THIS IS WHERE VERTEX NUMBERING HAPPENS
    # go over chars, get the row index of that char
    edge_beginnings = [df.index[df["key"] == c][0] for c in key_pairs[0]]
    edge_endings = [df.index[df["key"] == c][0] for c in key_pairs[1]]

    # now encode key as from letter_encoding (as int value, not python enum object)
    df["acc"] = df["key"].apply(lambda x: letter_encoding.has_accent(x))
    df["cap"] = df["key"].apply(lambda x: letter_encoding.is_capitalized(x))
    df["key"] = df["key"].apply(lambda x: char_to_value_map(x))

    return df, edge_beginnings, edge_endings, key_pairs_to_durs



char_map_func = letter_encoding.char_to_enum_value
# 81.4 ale bez labelek git
df = pd.read_csv(r'a.csv', sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE)

df = df[0:rows_per_example]


node_labels_df, edge_beginnings, edge_endings, edge_labels_info  = process_df(df, lambda x: x) 

print(node_labels_df)
print(edge_beginnings)
print(edge_endings)
print(edge_labels_info)



# Create a NetworkX graph
G = nx.DiGraph()  # Use nx.DiGraph() for directed graphs

# Add edges
edges = list(zip(edge_beginnings, edge_endings))
G.add_edges_from(edges)

# Map node indices to labels
node_mapping = {i: label for i, label in enumerate(node_labels_df["key"])}

reverse_node_mapping = {
    v: k for k, v in node_mapping.items()
}

# Apply reverse mapping to both values of key in edge_labels_info
edge_labels_info = {
    (reverse_node_mapping[src], reverse_node_mapping[dst]): label 
    for (src, dst), label in edge_labels_info.items()
}

nx.relabel_nodes(G, node_mapping, copy=False)

# Relabel the edge_labels_info to match relabeled nodes
edge_labels = {(node_mapping[src], node_mapping[dst]): label 
               for (src, dst), label in edge_labels_info.items()}

# Visualize the graph

pos = nx.shell_layout(G)  # Layout for visualization
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', arrows=True)

# Add edge labels
edge_label_positions = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', rotate=False)


# Dynamically adjust edge labels to avoid overlap
texts = [text for text in edge_label_positions.values()]
adjust_text(texts)


plt.title("Graph Visualization from Lists and DataFrame")
plt.show()