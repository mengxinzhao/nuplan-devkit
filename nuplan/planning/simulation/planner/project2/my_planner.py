import math
import logging
from typing import List, Type, Optional, Tuple, Dict
import click

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.planning.simulation.planner.project2.simple_predictor import SimplePredictor
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.planning.simulation.planner.project2.strategy import (
    HighSpeedStrategyConfiguration,
    LowSpeedStrategyConfiguration,
    HighSpeedLateralMovementStrategy,
    HIGH_SPEED_VELOCITY_THRESHOLD_MPS,
    VelocityKeepingLongitudinalMovementStrategy,
    CostWeight,
)  
from nuplan.planning.simulation.planner.project2.frenet import FrenetPath, cartesian_to_frenet

from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects

logger = logging.getLogger(__name__)


class FrenetOptimalTrajectoryPlanner(AbstractPlanner):
    """
    Simplified Optimal motion planning in frenet frame
    Paper:  Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
    ):
        """
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity
        self.target_velocity = 10
        self.max_accel = 3.0
        self.max_decel = 3.0
        self.min_turning_radius = 1.0
        self.max_curvature = 1.0 / self.min_turning_radius
        self.optimal_path = None

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False
        self.last_optimal_s, self.last_optimal_s_dot, self.last_optimal_s_2dot, self.last_optimal_t = None, None, None, None

        # Movement strategy used in the planning
        self._lateral_movement_strategy = HighSpeedLateralMovementStrategy()
        self._speed_profile_strategy = VelocityKeepingLongitudinalMovementStrategy()

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """

        # 1. Routing
        ego_state, observations = current_input.history.current_state
        if not self._routing_complete:
            self._router._initialize_ego_path(ego_state, self.max_velocity)
            self._routing_complete = True

        # 2. Generate reference line
        self._reference_path_provider = ReferenceLineProvider(self._router)
        self._reference_path_provider._reference_line_generate(ego_state)

        # 3. Objects prediction
        self._predictor = SimplePredictor(ego_state, observations, self.horizon_time.time_s, self.sampling_time.time_s)
        objects = self._predictor.predict()

        # 4. Planning
        trajectory: List[EgoState] = self.planning(ego_state, self._reference_path_provider, objects,
                                                    self.horizon_time, self.sampling_time, self.max_velocity, self.target_velocity)

        return InterpolatedTrajectory(trajectory)

    def _compute_lateral_offset(
            self, x_interp, y_interp, heading_interp, s, ego_x, ego_y
        ):
        """compute lateral offset by projecting Ego(x, y) to reference line. s is the projected point's arc """
        x_s = x_interp(s)
        y_s = y_interp(s)
        theta_s = heading_interp(s)
        offset_vec = np.array([ego_x - x_s, ego_y - y_s])
        perp = np.array([-np.sin(theta_s), np.cos(theta_s)])
        return np.dot(offset_vec, perp)

    def compute_frenet_path(
        self,
        target_speed: float,
        s0: float,  # s0
        s_d0: float,  # s'(t)
        s_dd0: float,  # s''(t)
        d0: float,
        d_d0: float,  # d'(s)
        d_dd0: float,  # d''(s))
    ) -> List[FrenetPath]:

        # calculate strategy_config
        if target_speed > HIGH_SPEED_VELOCITY_THRESHOLD_MPS and s_d0 > HIGH_SPEED_VELOCITY_THRESHOLD_MPS:
            strategy_config = HighSpeedStrategyConfiguration
        else:
            strategy_config = LowSpeedStrategyConfiguration
        path_candidates = []
        # T_i set to = horizon_
        t_i = self.horizon_time.time_s
        max_s = self._reference_path_provider._s_of_reference_line[-1] - 10  # Leave some margin
        t_i = min(t_i, max_s / target_speed) if target_speed > 0 else t_i
        lon_paths = self._speed_profile_strategy.calc_longitudinal_trajectory(
            target_speed,
            # start
            s0,
            # s'(t)
            s_d0,
            # s''(t)
            s_dd0,
            t_i,
            self.sampling_time.time_s,
            strategy_config,
        )
        for fp in lon_paths:
            # generate lateral trajectory for each speed profile
            for d_i in self._speed_profile_strategy.get_d_arrange(s0, self._reference_path_provider, strategy_config):

                updated_fp = self._lateral_movement_strategy.calc_lateral_trajectory(fp, d0, d_d0, d_dd0, d_i, t_i)

                # calculate cost
                Jp = sum(np.power(updated_fp.d_ddd, 2))  # square of lateral jerk
                Js = sum(np.power(updated_fp.s_ddd, 2))  # square of lateral jerk

                if target_speed >= HIGH_SPEED_VELOCITY_THRESHOLD_MPS:
                    # highspeed
                    lat_cost = CostWeight.K_J * Jp + CostWeight.K_T * t_i + CostWeight.K_D * updated_fp.d[-1] ** 2
                else:
                    # lowspeed
                    S = updated_fp.s[-1] - updated_fp.s[0]
                    lat_cost = CostWeight.K_J * Jp + CostWeight.K_T * S + CostWeight.K_D * updated_fp.d[-1] ** 2

                lon_cost = (
                    CostWeight.K_J * Js
                    + CostWeight.K_T * t_i
                    + self._speed_profile_strategy.calc_destination_cost(
                        target_speed, updated_fp, strategy_config
                    )
                )
                updated_fp.cf = CostWeight.K_LAT * lat_cost + CostWeight.K_LON * lon_cost
                path_candidates.append(updated_fp)
        return path_candidates

    def compute_transformed_path(self, candidate_paths: List[FrenetPath]):
        for i in range(len(candidate_paths)):
            # produce frenet -> cartesian transform
            candidate_paths[i] = self._lateral_movement_strategy.calc_cartesian_parameters(
                candidate_paths[i], self._reference_path_provider
            )
        return candidate_paths

    def check_collision(
        self,
        candidate_path: FrenetPath,
        tracked_objects: TrackedObjects,
    ):
        # TODO: axis seperation collision check
        # Check waypoint if x, y close to tracked objects within radius
        for tracked in tracked_objects:
            if not tracked.predictions:
                center_x = tracked.box.center.x
                center_y = tracked.box.center.y
                d_over_time = [
                    ((ix - center_x) ** 2 + (iy - center_y) ** 2)
                    for (ix, iy) in zip(candidate_path.x, candidate_path.y)
                ]
            else:
                waypoints:Waypoint  = tracked.predictions[0].waypoints  # PredictedTrajectory
                d_over_time = [
                    ((ix - wp._oriented_box.center.x) ** 2 + (iy - wp._oriented_box.center.y) ** 2)
                    for (wp, ix, iy) in zip(
                        waypoints,
                        candidate_path.x,
                        candidate_path.y,
                    )
                ]

            collision = any([d < 2 for d in d_over_time ])

            if collision: 
                return True

        return False

    def check_paths(
        self, candidates: List[FrenetPath], track_objects: TrackedObjects
    ) -> List[FrenetPath]:

        filtered_candidates = []

        for candidate in candidates:

            if any(np.array(candidate.v) > self.max_velocity):
                click.secho(f"candidate velocity > max_velocity", fg="red")
                continue
            if any(np.abs(candidate.a) > self.max_accel):
                click.secho(f"candidate accel > max_accel", fg="red")
                continue
            if any(np.abs(candidate.kappa) > self.max_curvature):
                click.secho(f"candidate curvature > max_curvature", fg="red")
                continue
            if self.check_collision(candidate, track_objects):
                click.secho(
                    f"candidate collides with one(or more) track_objects", fg="red"
                )
                continue
            filtered_candidates.append(candidate)
        return filtered_candidates

    def planning(
        self,
        ego_state: EgoState,
        reference_path_provider: ReferenceLineProvider,
        tracked_objects: TrackedObjects,
        horizon_time: TimePoint,
        sampling_time: TimePoint,
        max_velocity: float,
        target_velocity: float
    ) -> List[EgoState]:
        """
        Implement trajectory planning based on input and output, recommend using lattice planner or piecewise jerk planner.
        param: ego_state Initial state of the ego vehicle
        param: reference_path_provider Information about the reference path
        param: objects Information about dynamic obstacles
        param: horizon_time Total planning time
        param: sampling_time Planning sampling time
        param: max_velocity Planning speed limit (adjustable according to road speed limits during planning process)
        param: target_velocity Planning target speed
        return: trajectory Planning result
        """

        ego_pos = ego_state.rear_axle
        x_interp = reference_path_provider._interp1d_x
        y_interp = reference_path_provider._interp1d_y
        heading_interp = reference_path_provider._interp1d_heading
        kappa_interp = reference_path_provider._interp1d_kappa
        dkappa_interp = interp1d(
            reference_path_provider._s_of_reference_line,
            np.gradient(
                reference_path_provider._kappa_of_reference_line,
                reference_path_provider._s_of_reference_line,
            ),
        )

        def objective(s):
            # Compute the cost of a given lateral offset
            x_s = x_interp(s)
            y_s = y_interp(s)
            return (ego_pos.x - x_s) ** 2 + (ego_pos.y - y_s) ** 2

        result = minimize_scalar(
            objective,
            bounds=(
                reference_path_provider._s_of_reference_line[0],
                reference_path_provider._s_of_reference_line[-1],
            ),
            method="bounded",
        )

        s_ref = float(result.x)
        x_ref = x_interp(s_ref)
        y_ref = y_interp(s_ref)
        theta_ref = heading_interp(s_ref)
        kappa_ref = kappa_interp(s_ref)    
        dkappa_ref = dkappa_interp(s_ref)

        print(f"Found closest s_ref={s_ref:.2f} at tick")
        print(f"Ego position: ({ego_pos.x:.2f}, {ego_pos.y:.2f})")
        print(f"Ref position: ({x_ref:.2f}, {y_ref:.2f})")

        [s0, s_d0, s_dd0], [d0, d_d0, d_dd0] = cartesian_to_frenet(
            s_ref,
            x_ref,
            y_ref,
            theta_ref,
            kappa_ref,
            dkappa_ref,
            ego_pos.x,
            ego_pos.y,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.center.heading,
            math.tan(ego_state.tire_steering_angle)
            / ego_state.car_footprint.vehicle_parameters.wheel_base,
        )
        print(f"{s0=}, {s_d0=}, {s_dd0=}, {d0=}, {d_d0=}, {d_dd0=}, tire_steering_angle={ego_state.tire_steering_angle}")

        optimal_path = self.frenet_path_planning(self.target_velocity, s0, s_d0, s_dd0, d0, d_d0, d_dd0, tracked_objects)

        # 4.Produce ego trajectory
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            ),
            # tire_steering_angle=ego_state.dynamic_car_state.tire_steering_rate,
            tire_steering_angle=ego_state.tire_steering_angle,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )

        trajectory: List[EgoState] = [state]

        for iter in range(int(horizon_time.time_us / sampling_time.time_us)):
            relative_time = (iter + 1) * sampling_time.time_s
            # 根据relative_time 和 speed planning 计算 velocity accelerate （三次多项式）
            s, velocity, accelerate = cal_dynamic_state(
                relative_time,
                optimal_path.t,
                optimal_path.s,
                optimal_path.s_d,
                optimal_path.s_dd
            )
            # 根据当前时间下的s 和 路径规划结果 计算 x y heading kappa （线形插值）
            x, y, heading, _ = cal_pose(
                s,
                optimal_path.path_idx2s,
                optimal_path.x,
                optimal_path.y,
                optimal_path.yaw,
                optimal_path.kappa,
            )
            # steering_angle = math.atan(
            #     state.car_footprint.vehicle_parameters.wheel_base
            #     * optimal_path.kappa[min(iter, len(optimal_path.kappa) - 1)]
            # )
            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(velocity, 0),
                rear_axle_acceleration_2d=StateVector2D(accelerate, 0),
                tire_steering_angle=heading,
                time_point=state.time_point + sampling_time,
                vehicle_parameters=state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=0,
                angular_accel=0,
            )

            trajectory.append(state)

        return trajectory

    def frenet_path_planning(
        self,
        target_speed: float,
        s0: float, # s0
        s_d0: float, # s'(t)
        s_dd0: float, # s''(t)
        d0: float, 
        d_d0: float, # d'(s)
        d_dd0: float, # d''(s)
        tracked_objects: TrackedObjects
    ) -> FrenetPath:

        """Find the optimal path in Frenet frame given the reference line and return the path sample points in Frenet frame."""

        # TODO: update movement, speed_profile based on traffic observation and intention

        candidate_paths = self.compute_frenet_path(target_speed, s0, s_d0, s_dd0, d0, d_d0, d_dd0)
        candidate_paths = self.compute_transformed_path(candidate_paths)
        collision_free_paths = self.check_paths(candidate_paths, tracked_objects)
        print(f"num candidate path: {len(candidate_paths)}")
        print(f"num collision free path: {len(collision_free_paths)}")
        # find minimum cost path
        optimal_path = min(collision_free_paths, key=lambda fp: fp.cf, default=None)

        if optimal_path:
            print(f"optimal_path v: {optimal_path.v}")
            print(f"optimal_path s: {optimal_path.s}")
            self.optimal_path  = optimal_path
        else:
            print("optimal path is None!")
            # optimal_path = self.optimal_path
            # optimal_path.t =  optimal_path.t[:-1]
            # optimal_path.a = optimal_path.a[1:]
            # optimal_path.v = optimal_path.v[1:]
            # optimal_path.x = optimal_path.x[1:]
            # optimal_path.y = optimal_path.y[1:]
            # optimal_path.yaw = optimal_path.yaw[1:]
            # optimal_path.kappa = optimal_path.kappa[1:]
            # optimal_path.d = optimal_path.d[1:]
            # optimal_path.d_d = optimal_path.d_d[1:]
            # optimal_path.d_dd = optimal_path.d_dd[1:]
            # optimal_path.d_ddd = optimal_path.d_ddd[1:]
            # optimal_path.s = optimal_path.s[1:]
            # optimal_path.s_d = optimal_path.s_d[1:]
            # optimal_path.s_dd = optimal_path.s_dd[1:]
            # optimal_path.s_ddd = optimal_path.s_ddd[1:]
            # optimal_path.cf = optimal_path.cf

            # optimal_path.path_idx2s=optimal_path.path_idx2s[1:]

        return optimal_path
