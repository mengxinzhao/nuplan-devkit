import numpy as np
from typing import List, Type, Optional, Tuple
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.agent_state import StateSE2
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.planner.project2.abstract_predictor import (
    AbstractPredictor,
)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.agent import Agent
from nuplan.planning.simulation.trajectory.predicted_trajectory import (
    PredictedTrajectory,
)
from nuplan.common.actor_state.waypoint import Waypoint


class SimplePredictor(AbstractPredictor):
    def __init__(self, ego_state: EgoState, observations: Observation, duration: float, sample_time: float) -> None:
        self._ego_state = ego_state
        self._observations = observations
        self._duration = duration
        self._sample_time = sample_time
        self._num_samples = int(self._duration / self._sample_time)
        self._occupancy_map_radius = 40

    def predict(self):
        """Inherited, see superclass."""
        if isinstance(self._observations, DetectionsTracks):
            objects_init = self._observations.tracked_objects.tracked_objects
            objects = [
                object
                for object in objects_init
                if np.linalg.norm(self._ego_state.center.array - object.center.array) < self._occupancy_map_radius
            ]

            # Constant velocity model
            # TODO should use spline + prediction
            # probability = 1.0. trajectory length = duration / sample_time
            for object in objects:
                current_pose = object.center
                velocity = object.velocity
                # No steering
                heading_rate = 0
                waypoints = []  # List[Waypoint]
                for i in range(self._num_samples):
                    time_us = int((i + 1) * self._sample_time * 1e6)
                    delta_t = (i + 1) * self._sample_time
                    new_pos_x = current_pose.x + velocity.x * delta_t
                    new_pos_y = current_pose.y + velocity.y * delta_t
                    new_heading = current_pose.heading + heading_rate * delta_t
                    # Append to waypoints keep its original orientation box
                    waypoints.append(
                        Waypoint(
                            time_point=time_us,
                            oriented_box=OrientedBox.from_new_pose(
                                object.box, StateSE2(new_pos_x, new_pos_y, new_heading)
                            ),
                            velocity=velocity,
                        )
                    )
                # only one predictio
                predicted_trajectories = [PredictedTrajectory(
                    waypoints=waypoints, probability=1.0
                )] 
                object.predictions = predicted_trajectories

            return objects

        else:
            raise ValueError(
                f"SimplePredictor only supports DetectionsTracks. Got {self._observations.detection_type()}")
