import math
import logging
from typing import List, Type, Optional, Tuple, Dict

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

    def _compute_s_box(
        self,
        ego_length: float,
        ego_path_s: np.array,
        ego_path_x: np.array,
        ego_path_y: np.array,
        ego_heading: np.array,
        obj_x: float,
        obj_y: float,
        obj_length: float,
        obj_width: float,
        obj_heading: float,
        safety_buffer_front: float,
        safety_buffer_rear: float,

    ) -> Tuple[float, float, float]:
        """Compute the s box of track object in S-T graph and return l_offset, s_min, s_max
        """
        # Compute the s box of the object in the S-T graph
        l_offset = 0.0  # Placeholder for lateral offset
        s_min = 0.0  # Placeholder for minimum s value
        s_max = 0.0  # Placeholder for maximum s value

        # Compute distances to path points
        dists = np.hypot(ego_path_x - obj_x, ego_path_y - obj_y)
        i = np.argmin(dists)
        min_dist = dists[i]

        delta_theta = obj_heading - ego_heading[i]
        abs_cos_delta = np.abs(np.cos(delta_theta))
        abs_sin_delta = np.abs(np.sin(delta_theta))
        M = np.array([[abs_cos_delta, abs_sin_delta], [abs_sin_delta, abs_cos_delta]])
        half_dims = np.array([obj_length / 2, obj_width / 2])
        # half effective lon, lat of object after projecting itself into Ego's heading direction
        half_eff_long, half_eff_lat = M @ half_dims

        # Compute signed lateral offset l
        vec = np.array([obj_x - ego_path_x[i], obj_y - ego_path_y[i]])
        perp = np.array([-np.sin(ego_heading[i]), np.cos(ego_heading[i])])
        l = np.dot(vec, perp)

        s = ego_path_s[i]
        s_min = s - half_eff_long - ego_length / 2 - safety_buffer_rear
        s_max = s + half_eff_long + ego_length / 2 + safety_buffer_front

        return l_offset, s_min, s_max

    def _compute_st_boundaries(
        self,
        ego_state: EgoState,
        tracked_objects: TrackedObjects,
        path_x: List[float],
        path_y: List[float],
        path_heading: List[float],
        path_idx2s: List[float],
        time_stamps: List[float]
    ) -> List[Tuple[float, float]]:
        """Compute feasible ST region considering obstacles."""

        # in meters
        ego_length = ego_state.car_footprint.vehicle_parameters.length
        ego_width = ego_state.car_footprint.vehicle_parameters.width
        safety_buffer_front = 5.0  # safety distance in front
        safety_buffer_rear = 5.0  # safety distance behind
        lateral_threshold = 5.0  # lateral distance to consider obstacles/collision risk

        intervals: List[Tuple[float, float]] = []
        boundaries: List[Tuple[float, float]] = []

        path_x_np = np.array(path_x)
        path_y_np = np.array(path_y)
        path_heading_np = np.array(path_heading)
        path_idx2s_np = np.array(path_idx2s)

        for t_idx, tracked in enumerate(tracked_objects):
            if not tracked.predictions:
                l, s_min, s_max = self._compute_s_box(
                    ego_length,
                    path_idx2s_np,
                    path_x_np,
                    path_y_np,
                    path_heading_np,
                    tracked.box.center.x,
                    tracked.box.center.y,
                    tracked.box.length,
                    tracked.box.width,
                    tracked.box.heading,
                    safety_buffer_front,
                    safety_buffer_rear,
                )
                if abs(l) > lateral_threshold:
                    continue

                for t_idx in range(len(boundaries)):
                    boundaries[t_idx].append((s_min, s_max))
            else:
                # dynamic object
                prediction = tracked.predictions[0]  # PredictedTrajectory
                for t_idx, waypoint in enumerate(prediction.waypoints):
                    if t_idx >= len(boundaries):
                        break
                    l, s_min, s_max = self._compute_s_box(
                        ego_length,
                        path_idx2s_np,
                        path_x_np,
                        path_y_np,
                        path_heading_np,
                        waypoint.oriented_box.center.x,
                        waypoint.oriented_box.center.y,
                        waypoint.oriented_box.length,
                        waypoint.oriented_box.width,
                        waypoint.oriented_box.heading,
                        safety_buffer_front,
                        safety_buffer_rear,
                    )

                    if abs(l) > lateral_threshold:
                        continue

                    boundaries[t_idx].append((s_min, s_max))

        # Merge intervals for each time step
        for t_idx in range(len(boundaries)):
            intervals = boundaries[t_idx]
            if intervals:
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for current in intervals[1:]:
                    last = merged[-1]
                    if current[0] <= last[1]:
                        merged[-1] = (last[0], max(last[1], current[1]))
                    else:
                        merged.append(current)
                boundaries[t_idx] = merged

        return merged

    def _generate_quintic_speed_profile(
        self,
        s0: float,
        v0: float,
        a0: float,
        s_end: float,
        v_end: float,
        a_end: float,
        t_end: float,
        dt: float,
    ) -> Optional[dict]:
        """Generate quintic polynomial speed profile."""
        # Quintic polynomial: s(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        a0 = s0
        a1 = v0
        a2 = a0 / 2.0

        # Solve for remaining coefficients using boundary conditions
        t = t_end
        A = np.array(
            [
                [t**3, t**4, t**5],
                [3 * t**2, 4 * t**3, 5 * t**4],
                [6 * t, 12 * t**2, 20 * t**3],
            ]
        )
        b = np.array(
            [s_end - a0 - a1 * t - a2 * t**2, v_end - a1 - 2 * a2 * t, a_end - 2 * a2]
        )

        try:
            a3, a4, a5 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        # Generate profile
        num_points = int(t_end / dt) + 1
        t_vals = []
        s_vals = []
        v_vals = []
        a_vals = []

        for i in range(num_points):
            t = i * dt
            if t > t_end:
                t = t_end

            s = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
            v = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
            a = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3

            t_vals.append(t)
            s_vals.append(s)
            v_vals.append(v)
            a_vals.append(a)

        return {"t": t_vals, "s": s_vals, "v": v_vals, "a": a_vals}

    def _check_speed_profile_constraints(
        self, profile: dict, max_v: float, max_a: float, max_d: float,
        st_boundaries: List[Tuple[float, float]], s_max: float
    ) -> bool:
        """Check if speed profile satisfies all constraints."""
        for i, (s, v, a) in enumerate(zip(profile['s'], profile['v'], profile['a'])):
            # Velocity constraint
            if v < 0 or v > max_v:
                return False

            # Acceleration constraint
            if a < max_d or a > max_a:
                return False

            # Path length constraint
            if s < 0 or s > s_max:
                return False

            # ST boundary constraint
            for j in range(len(st_boundaries)):
                s_min, s_max_bound = st_boundaries[j]
                if  s_min <= s <= s_max_bound:
                    return False

        return True

    def _compute_speed_profile_cost(
        self, profile: dict, target_v: float, max_v: float
    ) -> float:
        """Compute cost for a speed profile."""
        # Cost components
        w_accel = 1.0  # Weight for acceleration
        w_jerk = 0.5   # Weight for jerk
        w_speed = 0.5  # Weight for speed tracking

        cost = 0.0

        # Acceleration cost (comfort)
        accel_cost = sum([a**2 for a in profile['a']])
        cost += w_accel * accel_cost

        # Jerk cost (smoothness)
        if len(profile['a']) > 1:
            dt = profile['t'][1] - profile['t'][0] if len(profile['t']) > 1 else 0.1
            jerk = [(profile['a'][i+1] - profile['a'][i])/dt 
                    for i in range(len(profile['a'])-1)]
            jerk_cost = sum([j**2 for j in jerk])
            cost += w_jerk * jerk_cost

        # Speed tracking cost (efficiency)
        speed_cost = sum([(v - target_v)**2 for v in profile['v']])
        cost += w_speed * speed_cost

        return cost

    def _generate_fallback_speed_profile(
        self, initial_v: float, s_max: float, horizon: float, dt: float
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Generate simple fallback speed profile."""
        num_points = int(horizon / dt) + 1
        t_vals = [i * dt for i in range(num_points)]

        # Simple constant velocity profile
        v_const = min(initial_v, self.max_velocity)
        s_vals = [min(v_const * t, s_max) for t in t_vals]
        v_vals = [v_const if s < s_max else 0.0 for s in s_vals]
        a_vals = [0.0] * num_points

        return s_vals, v_vals, a_vals, t_vals

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
                 tracked_objects: TrackedObjects,
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
            self.path_planning(ego_state, reference_path_provider)
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
            self.speed_planning(
                ego_state,
                horizon_time,
                sampling_time
                max_velocity,
                tracked_objects,
                path_idx2s,
                path_x,
                path_y,
                path_heading,
                path_kappa
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
        num_points = int(self.horizon_time.time_s/self.sampling_time.time_s) + 1
        # Output candidate path
        candidate_paths = []
        path_costs = []

        # weight for cost function
        w_offset = 1.0
        w_kappa = 5.0
        w_kappa_change = 10.0

        # Compute initial lateral offset (l_start) by projecting ego position onto reference path
        ego_pos = ego_state.center
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
        offset_vec = np.array([ego_pos.x - x_start, ego_pos.y - y_start])
        perp = np.array([-np.sin(theta_ref), np.cos(theta_ref)])
        l_start = np.dot(offset_vec, perp)

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
            for l_end in lateral_samples:
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
                b = np.array(
                    [
                        l_end - a0 - a1 * s - a2 * s**2,
                        dl_end - a1 - 2 * a2 * s,
                        ddl_end - 2 * a2,
                    ]
                )
                # Solve for coefficients
                a3, a4, a5 = np.linalg.solve(coeffs, b)
                path_l = []
                path_dl = []
                path_ddl = []
                path_s = []
                local_s_samples = np.linspace(0, s, num_points)
                for local_s in local_s_samples:
                    path_l.append(f(local_s))
                    path_dl.append(df(local_s))
                    path_ddl.append(ddf(local_s))
                    path_s.append(s_start + local_s)
                # Store the candidate path as a tuple (l, dl, ddl, s)
                candidate_paths.append((path_l, path_dl, path_ddl, path_s))

                path_costs.append(
                    w_offset * sum(np.array(path_l) ** 2)
                    + w_kappa * sum(np.array(path_dl) ** 2)
                    + w_kappa_change * sum(np.array(path_ddl) ** 2)
                )
                if path_costs[-1] < min_cost:
                    min_cost = path_costs[-1]
                    optimal_path_index = len(candidate_paths) - 1

        # TODO: Apollo after Lattice planner has DP programing to generate optimal path, feasible tunnel, nudging decision
        # and a spline QP solver to incorporate lane boundary constraints and dynamic feasibility to generate
        # a final smooth path
        # What I have is just Lattice Planner's optimal output
        return candidate_paths[optimal_path_index] if optimal_path_index >= 0 else ([], [], [], [])

    def speed_planning(
        self,
        ego_state: EgoState,
        max_velocity: float,
        tracked_objects: TrackedObjects,
        path_idx2s: List[float],
        path_x: List[float],
        path_y: List[float],
        path_heading: List[float],
        path_kappa: List[float],
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Speed planning using ST graph approach.
        
        Returns:
            optimal_speed_s: List of s-coordinates along the path
            optimal_speed_s_dot: List of velocities (ds/dt) at each point
            optimal_speed_s_2dot: List of accelerations (d²s/dt²) at each point
            optimal_speed_t: List of time stamps
        """
        # Vehicle dynamic constraints
        max_accel = 2.0  # m/s² - comfortable acceleration
        max_decel = -4.0  # m/s² - comfortable deceleration
        max_jerk = 2.0  # m/s³ - jerk limit for comfort

        s_max = path_idx2s[-1] if path_idx2s else 100.0
        s0 = 0.0
        v0 = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        a0 = ego_state.dynamic_car_state.rear_axle_acceleration_2d.magnitude()
        t_end = self.horizon_time.time_s
        dt  = self.sampling_time.time_s
        target_v = max_velocity

        # Generate time stamps
        num_points = int(t_end / dt) + 1
        time_stamps = [i * dt for i in range(num_points)]

        # Generate ST graph boundaries considering obstacles
        st_boundaries = self._compute_st_boundaries(
           ego_state, tracked_objects, path_x, path_y, path_heading, path_idx2s, time_stamps
        )
        # Sample end conditions
        v_ends = np.linspace(0, max_velocity, num=7)
        s_min_est = max(0, v0 * t_end + 0.5 * max_decel * t_end**2)
        s_max_est = v0 * t_end + 0.5 * max_accel * t_end**2
        s_ends = np.linspace(s_min_est, min(s_max_est, s_max), num=15)

        best_profile = None
        min_cost = np.inf
  
        # total sampled profiles are 7 * 15
        for v_end in v_ends:
            for s_end in s_ends:
                profile = self._generate_quintic_speed_profile(
                    s0, v0, a0, s_end, v_end, 0.0, t_end, dt
                )
                if profile is None:
                    continue
                if self._check_speed_profile_constraints(
                    profile, max_velocity, max_accel, max_decel, st_boundaries, s_max
                ):
                    cost = self._compute_speed_profile_cost(profile, target_v, max_velocity)
                    if cost < min_cost:
                        min_cost = cost
                        best_profile = profile

        if best_profile is None:
            optimal_s, optimal_s_dot, optimal_s_2dot, optimal_t = self._generate_fallback_speed_profile(
                v0, s_max, t_end, dt
            )
        else:
            optimal_s = best_profile['s']
            optimal_s_dot = best_profile['v']
            optimal_s_2dot = best_profile['a']
            optimal_t = best_profile['t']

        return optimal_s, optimal_s_dot, optimal_s_2dot, optimal_t