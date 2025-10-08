import json
import networkx as nx
from pyvis.network import Network

# Load JSON data
with open('description.json', 'r') as f:
    segments = json.load(f)

# Build the graph with networkx first
G = nx.DiGraph()

for i, segment in enumerate(segments):
    scene_id = f"scene_{segment['timestamp']}"
    G.add_node(scene_id, label=f"Scene {segment['timestamp']}", type='scene')

    if i > 0:
        prev_scene_id = f"scene_{segments[i-1]['timestamp']}"
        G.add_edge(prev_scene_id, scene_id, label='temporal')

    for action in segment.get('actions', []):
        action_id = f"action_{action}"
        if not G.has_node(action_id):
            G.add_node(action_id, label=action, type='action')
        G.add_edge(scene_id, action_id, label='scene-action')

# Use pyvis to render as HTML
net = Network(height='800px', width='100%', directed=True, notebook=True)

# Convert from networkx → pyvis
for node, data in G.nodes(data=True):
    node_type = data.get('type', 'other')
    color = 'skyblue' if node_type == 'scene' else 'lightgreen'
    net.add_node(node, label=data.get('label', node), color=color, title=node_type)

for u, v, data in G.edges(data=True):
    net.add_edge(u, v, label=data.get('label', ''))

# Enable physics + scrolling
net.toggle_physics(True)

# Save and open
net.show('scene_action_temporal_graph.html')

