import math
import logging
from typing import List, Type, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
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

from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.planning.simulation.planner.project2.frame_transform import get_match_point

logger = logging.getLogger(__name__)


class MyPlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
    ):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False

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
                                                    self.horizon_time, self.sampling_time, self.max_velocity)

        return InterpolatedTrajectory(trajectory)

    def planning(self,
                 ego_state: EgoState,
                 reference_path_provider: ReferenceLineProvider,
                 objects: List[TrackedObjects],
                 horizon_time: TimePoint,
                 sampling_time: TimePoint,
                 max_velocity: float) -> List[EgoState]:
        """
        Implement trajectory planning based on input and output, recommend using lattice planner or piecewise jerk planner.
        param: ego_state Initial state of the ego vehicle
        param: reference_path_provider Information about the reference path
        param: objects Information about dynamic obstacles
        param: horizon_time Total planning time
        param: sampling_time Planning sampling time
        param: max_velocity Planning speed limit (adjustable according to road speed limits during planning process)
        return: trajectory Planning result
        """
        # Optimization based planner
        # 1.Path planning
        optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s = (
            path_planning(ego_state, reference_path_provider)
        )

        # 2.Transform path planning result to cartesian frame
        path_idx2s, path_x, path_y, path_heading, path_kappa = transform_path_planning(
            optimal_path_s,
            optimal_path_l,
            optimal_path_dl,
            optimal_path_ddl,
            reference_path_provider,
        )

        # 3.Speed planning
        optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t = (
            speed_planning(
                ego_state,
                horizon_time.time_s,
                max_velocity,
                objects,
                path_idx2s,
                path_x,
                path_y,
                path_heading,
                path_kappa,
            )
        )

        # 4.Produce ego trajectory
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            ),
            tire_steering_angle=ego_state.dynamic_car_state.tire_steering_rate,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for iter in range(int(horizon_time.time_us / sampling_time.time_us)):
            relative_time = (iter + 1) * sampling_time.time_s
            # 根据relative_time 和 speed planning 计算 velocity accelerate （三次多项式）
            s, velocity, accelerate = cal_dynamic_state(
                relative_time,
                optimal_speed_t,
                optimal_speed_s,
                optimal_speed_s_dot,
                optimal_speed_s_2dot,
            )
            # 根据当前时间下的s 和 路径规划结果 计算 x y heading kappa （线形插值）
            x, y, heading, _ = cal_pose(
                s, path_idx2s, path_x, path_y, path_heading, path_kappa
            )

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

    def path_planning(self, ego_state: EgoState, reference_path_provider: ReferenceLineProvider) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Find the optimal path in Frenet frame given the reference line and return the path sample points in Frenet frame.

        TODO: 
        - Consider collision avoidance and its cost
        - Consider lane deviation and its lane change ost
        """
        # Extract reference path information
        s_ref = reference_path_provider._s_of_reference_line
        x_ref = reference_path_provider._x_of_reference_line
        y_ref = reference_path_provider._y_of_reference_line
        heading_ref = reference_path_provider._heading_of_reference_line
        s_total = s_ref[-1][0]  # Total longitudinal distance
        num_points = int(self.horizon_time/self.sampling_time)
        # Output candidate path
        candidate_paths = []
        path_costs = []

        # weight for cost function
        w_offset = 1.0
        w_kappa = 1.0
        w_kappa_change = 1.0

        # Compute initial lateral offset (l_start) by projecting ego position onto reference path
        ego_pos = ego_state.center
        match_index= get_match_point([ego_pos.x],[ego_pos.y], reference_path_provider._x_of_reference_line, reference_path_provider._y_of_reference_line)[0]
        # Interpolate reference path for x(s) and y(s)
        x_interp = interp1d(s_ref, x_ref, kind='cubic', fill_value='extrapolate')
        y_interp = interp1d(s_ref, y_ref, kind='cubic', fill_value='extrapolate')
        heading_interp = interp1d(s_ref, heading_ref, kind='linear', fill_value='extrapolate')

        # use minimize solver to find the closest point on reference line
        # TODO: this doesn't guarantee the trajectory planning consistency in respect to last planned trajectory's ending point
        def objective(s):
            # Compute the cost of a given lateral offset
            x_s = x_interp(s)
            y_s = y_interp(s)
            return (ego_pos.x - x_s)**2 + (ego_pos.y - y_s)**2

        result = minimize_scalar(objective, bounds=(s_ref[0], s_ref[-1]), method='bounded')
        s_start = float(result.x)
        x_start = x_interp(s_start)
        y_start = y_interp(s_start)

        # Compute heading for the planning start point
        ds = 0.01  # 0.01m arc delta
        s_head = min(max(s_ref[0], s_start + ds), s_ref[-1])
        x_head = float(x_interp(s_head))
        y_head = float(y_interp(s_head))
        theta_ref = math.atan2(y_head - y_start, x_head - x_start)

        # Compute lateral offset (l_start)
        dx = ego_pos.x - x_start
        dy = ego_pos.y - y_start
        l_start = dx * math.sin(theta_ref) - dy * math.cos(theta_ref)

        # Compute initial derivatives dl/ds and d2l/ds2
        s_prev = max(s_ref[0], s_start - ds)
        s_next = min(s_ref[-1], s_start + ds)
        theta_prev = float(heading_interp(s_prev))
        theta_next = float(heading_interp(s_next))
        kappa_ref = (theta_next - theta_prev) / (s_next - s_prev)

        # Approximate for small angles
        wheel_base = ego_state.car_footprint.vehicle_parameters.wheel_base
        kappa_ego = math.tan(ego_state.dynamic_car_state.tire_steering_angle) / wheel_base if wheel_base > 0 else 0.0
        # dl/ds  ≈  Δθ ~ ego_heading - ref_heading
        dl_start = ego_state.center.heading - theta_ref
        # d²l/ds² ≈ Δκ
        d2l_start = kappa_ego - kappa_ref

        # Define lattice parameters
        max_lateral_offset = 2.0  # Max lateral deviation (m)
        lateral_samples = np.linspace(l_start-max_lateral_offset, l_start + max_lateral_offset, 7)  # Sample 7 lateral offsets
        # TODO: longtitude sample can't exceed max velocity * horizon_time
        s_samples = [10, 20, 40, 80]
        min_cost = np.inf

        # Generate candidate paths for each lateral offset
        # Quintic polynomial: f(s) = a5 s^5 + a4 s^4 + a3 s^3 + a2 s^2 + a1 s + a0
        def f(s):
            return (a5 * s**5 + a4 * s**4 + a3 * s**3 + a2 * s**2 + a1 * s + a0)

        def df(s):
            return (5 * a5 * s**4 + 4 * a4 * s**3 + 3 * a3 * s**2 + 2 * a2 * s + a1)

        def ddf(s):
            return (20 * a5 * s**3 + 12 * a4 * s**2 + 6 * a3 * s + 2 * a2)

        optimal_path_index = -1
        for s in s_samples:
            if s > s_total - s_start:
                # exceeds reference path already
                continue
            for lat in lateral_samples:
                l_end = l_start + lat
                dl_end = 0.0
                ddl_end = 0.0
                # Fit Quintic polynomial: l(s) = a5 s^5 + a4 s^4 + a3 s^3 + a2 s^2 + a1 s + a0
                # on s = [0, lon]
                a0 = l_start
                a1 = dl_start
                a2 = d2l_start / 2.0
                coeffs = np.array(
                    [
                        [s**3, s**4, s**5],
                        [3 * s**2, 4 * s**3, 5 * s**4],
                        [6 * s, 12 * s**2, 20 * s**3],
                    ]
                )
                rhs = np.array(
                    [
                        l_end - a0 - a1 * s - a2 * s**2,
                        dl_end - a1 - 2 * a2 * s,
                        ddl_end - 2 * a2,
                    ]
                )
                # Solve for coefficients
                a3, a4, a5 = np.linalg.solve(coeffs, rhs)
                path_l = []
                path_dl = []
                path_ddl = []
                local_s_samples = np.linspace(0, s, num_points)
                for local_s in local_s_samples:
                    path_l.append(f(local_s))
                    path_dl.append(df(local_s))
                    path_ddl.append(ddf(local_s))

                path_s = [s_start + local_s for local_s in local_s_samples]
                # Store the candidate path as a tuple (l, dl, ddl, s)
                candidate_paths.append((path_l, path_dl, path_ddl, path_s))

                path_costs.append(
                    w_offset * sum(np.array(path_l) ** 2)
                    + w_kappa * sum(np.array(path_dl) ** 2)
                    + w_kappa_change * sum(path_ddl) ** 2
                )
                if path_costs[-1] < min_cost:
                    min_cost = path_costs[-1]
                    optimal_path_index = len(candidate_paths) - 1

        return candidate_paths[optimal_path_index] if optimal_path_index >= 0 else ([], [], [], [])
