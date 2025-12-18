from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import simpy


@dataclass
class Vehicle:
    """
    Phase 1 entity: a single-capacity vehicle that can execute one delivery at a time.

    Important rule: no dispatch policy in this class. The simulator decides what to do;
    the vehicle just simulates travel/service time.
    """

    vehicle_id: str
    capacity: int
    start_node: str
    env: simpy.Environment = field(repr=False)

    node_id: str = field(init=False)
    busy: simpy.Resource = field(init=False, repr=False)
    last_available_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.node_id = self.start_node
        self.busy = simpy.Resource(self.env, capacity=1)

    def is_idle(self) -> bool:
        return self.busy.count == 0 and len(self.busy.queue) == 0

    def deliver(
        self,
        *,
        pickup_node: str,
        dropoff_node: str,
        travel_time_to_pickup: float,
        travel_time_to_dropoff: float,
        service_time: float = 0.0,
    ) -> simpy.events.Event:
        """
        Returns a SimPy process that:
        - waits for vehicle availability
        - travels to pickup
        - travels to dropoff
        - optional service time
        - updates vehicle location
        """

        def _proc() -> simpy.events.Event:
            with self.busy.request() as req:
                yield req
                # Travel to pickup
                if travel_time_to_pickup > 0:
                    yield self.env.timeout(travel_time_to_pickup)
                self.node_id = pickup_node
                # Travel to dropoff
                if travel_time_to_dropoff > 0:
                    yield self.env.timeout(travel_time_to_dropoff)
                self.node_id = dropoff_node
                if service_time > 0:
                    yield self.env.timeout(service_time)
                self.last_available_time = float(self.env.now)

        return self.env.process(_proc())

