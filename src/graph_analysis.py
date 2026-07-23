"""
Graph Neural Network (GNN) analysis of social propagation patterns
to detect structural biases or bot networks spreading misinformation.

No real social-sharing/propagation data is available for this project
(CLAUDE.md: "graph analysis (optional/advanced) - if social-sharing data
is available"). Everything here is real, working code, but the dataset
functions below build clearly-labeled SYNTHETIC propagation cascades so
the GNN and the structural heuristic can be demonstrated and tested.
graph_to_data() is the integration point for a real retweet/share graph
loader once that data exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

NODE_FEATURE_NAMES = ["degree", "clustering", "in_out_ratio", "reciprocity"]


# ---------------------------------------------------------------------------
# Structural feature engineering + graph <-> PyG Data conversion
# ---------------------------------------------------------------------------


def extract_node_features(graph: nx.DiGraph) -> np.ndarray:
    """Per-node structural features from a propagation graph, where edge
    u->v means u's share/post was reshared by v.
    """
    undirected = graph.to_undirected()
    degrees = dict(graph.degree())
    clustering = nx.clustering(undirected)
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())

    features = []
    for node in graph.nodes():
        indeg, outdeg = in_deg[node], out_deg[node]
        in_out_ratio = indeg / (outdeg + 1)
        mutual = sum(1 for _, v in graph.out_edges(node) if graph.has_edge(v, node))
        reciprocity = mutual / outdeg if outdeg else 0.0
        features.append([degrees[node], clustering[node], in_out_ratio, reciprocity])
    if not features:
        return np.empty((0, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def graph_to_data(graph: nx.DiGraph, label: int | None = None) -> Data:
    """networkx DiGraph -> torch_geometric Data (node features + edge_index)."""
    nodes = list(graph.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    x = torch.tensor(extract_node_features(graph), dtype=torch.float)
    edges = [[index[u], index[v]] for u, v in graph.edges()]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = Data(x=x, edge_index=edge_index)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data


# ---------------------------------------------------------------------------
# Interpretable structural heuristic (supporting signal, not ML) - mirrors
# src/reasoning.py's rule-based checks, applied to graph topology instead
# of article text.
# ---------------------------------------------------------------------------


@dataclass
class BotNetworkSignal:
    triggered: bool
    score: float
    explanation: str


def flag_bot_network_signals(
    graph: nx.DiGraph,
    density_threshold: float = 0.5,
    reciprocity_threshold: float = 0.5,
    min_core_size: int = 3,
) -> BotNetworkSignal:
    """Coordinated/bot-driven sharing tends to include a small, unusually
    dense and reciprocal cluster (accounts mutually amplifying each
    other) that blasts out to many one-off peripheral nodes; organic
    sharing is typically sparse and tree-like throughout. Whole-graph
    density dilutes away as peripheral nodes pile up, so this looks at
    the densest embedded subgraph (max k-core) instead of the graph as a
    whole - a standard technique for finding a tight cluster hiding
    inside a larger sparse network. A supporting signal alongside the
    GNN's prediction, not a replacement for it.
    """
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return BotNetworkSignal(False, 0.0, "graph too small/sparse to assess")

    undirected = nx.Graph(graph)
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    core_numbers = nx.core_number(undirected)
    max_core = max(core_numbers.values())
    core_nodes = [node for node, c in core_numbers.items() if c == max_core]

    if len(core_nodes) < min_core_size:
        return BotNetworkSignal(False, 0.0, f"densest core has only {len(core_nodes)} node(s), too small to assess")

    core = graph.subgraph(core_nodes)
    density = nx.density(core)
    edges = core.number_of_edges()
    mutual = sum(1 for u, v in core.edges() if core.has_edge(v, u))
    reciprocity = mutual / edges if edges else 0.0

    triggered = density > density_threshold and reciprocity > reciprocity_threshold
    score = min((density / density_threshold + reciprocity / reciprocity_threshold) / 2, 1.0)
    return BotNetworkSignal(
        triggered=triggered,
        score=score,
        explanation=(
            f"densest core: {len(core_nodes)} nodes, density={density:.2f}, reciprocity={reciprocity:.2f} "
            f"({'above' if triggered else 'below'} bot-like thresholds)"
        ),
    )


# ---------------------------------------------------------------------------
# GNN: graph-level classifier (organic vs. coordinated/bot-driven cascade)
# ---------------------------------------------------------------------------


class PropagationGNN(nn.Module):
    """Embeds each node from its structural features, message-passes over
    the share/reshare edges, then pools to one organic-vs-bot-driven
    prediction per cascade.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(self.dropout(x))


# ---------------------------------------------------------------------------
# Synthetic propagation cascades (see module docstring)
# ---------------------------------------------------------------------------


def _simulate_organic_cascade(n_nodes: int, seed: int) -> nx.DiGraph:
    """Sparse, branching broadcast cascade - each share reaches a few new
    people, little mutual reciprocity."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_node(0)
    for node in range(1, n_nodes):
        parent = int(rng.integers(0, node))
        graph.add_edge(parent, node)
    return graph


def _simulate_bot_cascade(n_nodes: int, seed: int, core_fraction: float = 0.4) -> nx.DiGraph:
    """A small, densely and mutually connected core amplifying itself,
    blasting out to a few peripheral nodes - a known coordinated
    inauthentic behavior signature (high density, high reciprocity)."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    core_size = min(n_nodes, max(3, int(n_nodes * core_fraction)))
    core = list(range(core_size))
    # Add core nodes explicitly - add_edge() alone would only create a node
    # once it gets an edge, so a too-small or (rarely, by chance) edge-less
    # core would otherwise silently vanish from the graph instead of
    # appearing as isolated nodes.
    graph.add_nodes_from(core)
    for u in core:
        for v in core:
            if u != v and rng.random() < 0.7:
                graph.add_edge(u, v)
    for node in range(core_size, n_nodes):
        source = int(rng.choice(core))
        graph.add_edge(source, node)
    return graph


def build_synthetic_dataset(n_graphs: int = 60, min_nodes: int = 15, max_nodes: int = 30, seed: int = 0) -> list[Data]:
    rng = np.random.default_rng(seed)
    dataset = []
    for i in range(n_graphs):
        n_nodes = int(rng.integers(min_nodes, max_nodes + 1))
        if i % 2 == 0:
            graph, label = _simulate_organic_cascade(n_nodes, seed=seed + i), 0
        else:
            graph, label = _simulate_bot_cascade(n_nodes, seed=seed + i), 1
        dataset.append(graph_to_data(graph, label=label))
    return dataset


def main() -> None:
    """Demo: train PropagationGNN on synthetic cascades, report held-out
    accuracy, and compare the rule-based signal on one organic vs. one
    bot-driven example."""
    from sklearn.model_selection import train_test_split

    dataset = build_synthetic_dataset(n_graphs=80)
    labels = [int(d.y.item()) for d in dataset]
    train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=0, stratify=labels)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    model = PropagationGNN(in_channels=len(NODE_FEATURE_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.y.size(0)
    accuracy = correct / total
    print(f"Synthetic held-out accuracy: {correct}/{total} = {accuracy:.2%}")
    if accuracy == 1.0:
        print(
            "  (expected here: the two synthetic generators produce structurally "
            "distinct graphs by construction - tree-like vs. dense-core-plus-blast. "
            "This is a mechanics check, not a claim about real propagation data, "
            "which won't separate this cleanly. Per CLAUDE.md: a perfect score on "
            "real-world data would need a leakage investigation before trusting it.)"
        )

    organic = _simulate_organic_cascade(20, seed=999)
    bot = _simulate_bot_cascade(20, seed=999)
    print("organic cascade signal:", flag_bot_network_signals(organic))
    print("bot cascade signal:    ", flag_bot_network_signals(bot))


if __name__ == "__main__":
    main()
