"""Street graph container (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Graph:
    """Undirected weighted graph with optional step-free flags on edges."""

    adj: Dict[int, List[Tuple[int, float, bool]]] = field(default_factory=dict)
    node_xy: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    def add_edge(
        self, a: int, b: int, length_m: float, step_free: bool = True
    ) -> None:
        self.adj.setdefault(a, []).append((b, length_m, step_free))
        self.adj.setdefault(b, []).append((a, length_m, step_free))

    def neighbors(
        self, node: int, require_step_free: bool = False
    ) -> List[Tuple[int, float]]:
        out = []
        for nbr, length, step_free in self.adj.get(node, []):
            if require_step_free and not step_free:
                continue
            out.append((nbr, length))
        return out
