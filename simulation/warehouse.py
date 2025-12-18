from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Warehouse:
    """
    Phase 1 entity: holds inventory at a graph node.

    Important rule: no decision-making in this class.
    """

    warehouse_id: str
    node_id: str
    inventory: int

    def can_fulfill(self, qty: int) -> bool:
        return self.inventory >= qty

    def allocate(self, qty: int) -> bool:
        """
        Atomically decrements inventory if possible.
        Returns True if allocated, False if stockout.
        """
        if qty <= 0:
            return True
        if self.inventory >= qty:
            self.inventory -= qty
            return True
        return False

    def restock(self, qty: int) -> None:
        if qty <= 0:
            return
        self.inventory += qty

