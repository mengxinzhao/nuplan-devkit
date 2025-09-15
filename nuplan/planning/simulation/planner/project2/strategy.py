from enum import Enum, auto
from dataclasses import dataclass, field
from nuplan.planning.simulation.planner.project2.polynomial import (
    QuarticPolynomial,
    QuinticPolynomial,
)
from nuplan.planning.simulation.planner.project2.merge_path_speed import (
    transform_path_planning,
)
from nuplan.planning.simulation.planner.project2.reference_line_provider import (
    ReferenceLineProvider,
)
from nuplan.planning.simulation.planner.project2.frenet import FrenetPath
from typing import Tuple, List, Union
import numpy as np
import bisect
import math


class LateralMovement(Enum):
    HIGH_SPEED = auto()
    LOW_SPEED = auto()


class LongitudinalMovement(Enum):
    MERGING_AND_STOPPING = auto()
    VELOCITY_KEEPING = auto()


# Thresholds to consider high speed 60mph
HIGH_SPEED_VELOCITY_THRESHOLD_MPS = 26.8224


@dataclass
class CostWeight:
    K_J: float = 0.1
    K_T: float = 0.1
    # used in velocity keep
    K_S_DOT: float = 1.0
    # cost weight on lateral offset
    K_D: float = 1.0
    # cost weight on longitudinal progress
    K_S: float = 1.0
    # cost weight on lateral movement
    K_LAT: float = 1.0
    # cost weight on longitudinal movement
    K_LON: float = 1.0


@dataclass
class LowSpeedStrategyConfiguration:
    width_sampling_m: float = field(default=0.2)
    # 2.2 mph a sample interval
    speed_sampling_mps: float = field(default=0.5)
    num_speed_samples: int = 10
    stop_distance_m: float = field(default=5.0)
    # stop point sampling length
    stop_sampling_m: float = field(default=0.3)
    # stop point samples
    num_stop_samples: int = field(default=4)


@dataclass
class HighSpeedStrategyConfiguration:
    width_sampling_m: float = field(default=0.25)
    # 5 mph a sample interval
    speed_sampling_mps: float = field(default=2.2352)
    num_speed_samples: int = 10
    stop_distance_m: float = field(default=25.0)
    # stop point sampling length
    stop_sampling_m: float = field(default=2.0)
    # stop point samples
    num_stop_samples: int = field(default=4)


class LateralMovementStrategy:
    def calc_lateral_trajectory(self, fp, c_d, c_d_d, c_d_dd, di, Ti):
        """
        Calculate the lateral trajectory
        """
        raise NotImplementedError("calc_lateral_trajectory not implemented")

    def calc_cartesian_parameters(
        self, frenet_path: FrenetPath, reference_line_provider: ReferenceLineProvider
    ) -> FrenetPath:
        """
        Calculate the cartesian parameters (x, y, yaw, curvature, v, a) from frenet path and
        cubic spline planner as reference line provider

        """
        (
            frenet_path.path_idx2s,
            frenet_path.x,
            frenet_path.y,
            frenet_path.yaw,
            frenet_path.kappa,
        ) = transform_path_planning(
            path_s=frenet_path.s,
            path_l=frenet_path.d,
            path_dl=frenet_path.d_d,
            path_ddl=frenet_path.d_dd,
            reference_path_provider=reference_line_provider,
        )

        ref_theta = reference_line_provider._interp1d_heading(frenet_path.s)
        ref_kappa = reference_line_provider._interp1d_kappa(frenet_path.s)
        ref_dkappa = np.gradient(ref_kappa, frenet_path.s)
        one_minus_kappa_r_d = 1 - np.array(ref_kappa) * np.array(frenet_path.d)

        d_dot = np.array(frenet_path.d_d) * np.array(frenet_path.s_d)
        frenet_path.v = np.sqrt(
            np.power(one_minus_kappa_r_d, 2) * np.power(frenet_path.s_d, 2)
            + np.power(d_dot, 2)
        )
        delta_theta = np.arctan2(frenet_path.d_d, one_minus_kappa_r_d)
        cos_delta_theta = np.cos(delta_theta)
        theta = np.arctan2(np.sin(delta_theta + ref_theta), np.cos(delta_theta + ref_theta))

        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * frenet_path.kappa - ref_kappa
        kappa_r_d_prime = (ref_dkappa) * frenet_path.d + ref_kappa * d_dot

        frenet_path.a = (
            frenet_path.s_dd * one_minus_kappa_r_d / cos_delta_theta
            + np.power(frenet_path.s_d, 2) / cos_delta_theta * (frenet_path.d_d * delta_theta_prime - kappa_r_d_prime)
        )
        # Print the two components separately
        print(f"Component 1 (s_dd term) first 5: {(frenet_path.s_dd * one_minus_kappa_r_d / cos_delta_theta)[:5]}")
        print(f"Component 2 (s_d^2 term) first 5: {(np.power(frenet_path.s_d, 2) / cos_delta_theta * (frenet_path.d_d * delta_theta_prime - kappa_r_d_prime))[:5]}")

        return frenet_path


class HighSpeedLateralMovementStrategy(LateralMovementStrategy):
    def calc_lateral_trajectory(
        self, fp: FrenetPath, d0: float, d_d0: float, d_dd0: float, di: float,  Ti: float
    ) -> FrenetPath:
        # why deepcopy?
        # tp = copy.deepcopy(fp)
        s_d0 = fp.s_d[0]
        s_dd0 = fp.s_dd[0]
        # d'(t) = d'(s) * s'(t)
        # d''(t) = d''(s) * s'(t)^2 + d'(s) * s''(t)
        # end point d_d1 = d_dd1 = 0
        # lateral trajectory in respect to t
        lat_qp = QuinticPolynomial(
            d0, d_d0 * s_d0, d_dd0 * s_d0**2 + d_d0 * s_dd0, di, 0.0, 0.0, Ti
        )
        # reset frenet_path lateral values
        fp.d = []
        fp.d_d = []
        fp.d_dd = []
        fp.d_ddd = []

        # Calculate all derivatives in a single loop to reduce iterations
        for i in range(len(fp.t)):
            t = fp.t[i]
            s_d = fp.s_d[i]
            s_dd = fp.s_dd[i]

            s_d_inv = 1.0 / (s_d + 1e-6) + 1e-6  # Avoid division by zero
            s_d_inv_sq = s_d_inv * s_d_inv  # Square of inverse

            d = lat_qp.interpolate(t)
            d_d = lat_qp.interpolate_derivative(t)
            d_dd = lat_qp.interpolate_second_derivative(t)
            d_ddd = lat_qp.interpolate_third_derivative(t)

            fp.d.append(d)
            # d'(s) = d'(t) / s'(t)
            fp.d_d.append(d_d * s_d_inv)
            # d''(s) = (d''(t) - d'(s) * s''(t)) / s'(t)^2
            fp.d_dd.append((d_dd - fp.d_d[i] * s_dd) * s_d_inv_sq)
            fp.d_ddd.append(d_ddd)
        return fp


class LowSpeedLateralMovementStrategy(LateralMovementStrategy):
    def calc_lateral_trajectory(
        self, fp: FrenetPath, d0: float, d_d0: float, d_dd0: float, di: float,  Ti: float
    ) -> FrenetPath:
        s0 = fp.s[0]
        s1 = fp.s[-1]
        # d = d(s), d_d = d'(s), d_dd = d''(s)
        # x(s(t), d(t)) = r(s(t)) + d(s(t)) * n_r (s(t))
        # lateral trajectory in respect to arch s
        lat_qp = QuinticPolynomial(d0, d_d0, d_dd0, di, 0.0, 0.0, s1 - s0)

        fp.d = [lat_qp.interpolate(s - s0) for s in fp.s]
        fp.d_d = [lat_qp.interpolate_derivative(s - s0) for s in fp.s]
        fp.d_dd = [lat_qp.interpolate_second_derivative(s - s0) for s in fp.s]
        fp.d_ddd = [lat_qp.interpolate_third_derivative(s - s0) for s in fp.s]
        return fp


class LongitudinalMovementStrategy:

    def calc_longitudinal_trajectory(
        self,
        target_speed: float,
        s_d0: float,
        s_dd0: float,
        s0: float,
        Ti: float,
        sampling_time: float,
        strategy_config: Union[
            LowSpeedStrategyConfiguration, HighSpeedStrategyConfiguration
        ],
    ) -> List[FrenetPath]:
        """
        Calculate the longitudinal trajectory
        """
        raise NotImplementedError("calc_longitudinal_trajectory not implemented")

    def get_d_arrange(
        self,
        s0: float,
        reference_line_provider: ReferenceLineProvider,
        strategy_config: Union[
            LowSpeedStrategyConfiguration, HighSpeedStrategyConfiguration
        ],
    ):
        """
        Get the lateral offset sample range
        """
        # localize s0 in reference_line
        s0_index = bisect.bisect(reference_line_provider._s_of_reference_line, s0)
        left_bound = reference_line_provider._lb_of_reference_line[s0_index]
        right_bound = reference_line_provider._rb_of_reference_line[s0_index]
        # have to count for half body width
        return np.arange(-right_bound + 1, left_bound - 1, strategy_config.width_sampling_m)

    def calc_destination_cost(
        self,
        target_speed,
        fp,
        strategy_config: Union[
            LowSpeedStrategyConfiguration, HighSpeedStrategyConfiguration
        ],
    ):
        """
        Calculate the destination cost
        """
        raise NotImplementedError("calc_destination_cost not implemented")


class VelocityKeepingLongitudinalMovementStrategy(LongitudinalMovementStrategy):
    def calc_longitudinal_trajectory(
        self,
        target_speed: float,
        s0: float,
        s_d0: float,
        s_dd0: float,
        Ti: float,
        sampling_time: float,
        strategy_config: Union[
            LowSpeedStrategyConfiguration, HighSpeedStrategyConfiguration
        ],
    ) -> List[FrenetPath]:
        fplist = []
        for target_v in np.arange(
            max(0, target_speed - strategy_config.speed_sampling_mps * strategy_config.num_speed_samples),
            target_speed + strategy_config.speed_sampling_mps * strategy_config.num_speed_samples,
            strategy_config.speed_sampling_mps,
        ):
            fp = FrenetPath()
            lon_qp = QuarticPolynomial(s0, s_d0, s_dd0, target_v, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, sampling_time)]
            fp.s = [lon_qp.interpolate(t) for t in fp.t]
            fp.s_d = [lon_qp.interpolate_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.interpolate_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.interpolate_third_derivative(t) for t in fp.t]
            fplist.append(fp)
            # print(f"{target_v=}\nfp.s_dd: {fp.s_dd}")
        return fplist

    def calc_destination_cost(
        self,
        target_speed: float,
        fp: FrenetPath,
        strategy_config: Union[
            LowSpeedStrategyConfiguration, HighSpeedStrategyConfiguration
        ],
    ):
        ds = (target_speed - fp.s_d[-1]) ** 2
        return CostWeight.K_S_DOT * ds


# TODO: Merging strategy
class StoppingLongitudinalMovementStrategy(LongitudinalMovementStrategy):
    def calc_longitudinal_trajectory(
        self,
        target_speed: float,
        s0: float,
        s_d0: float,
        s_dd0: float,
        Ti: float,
        sampling_time: float,
        strategy_config: Union[
            LowSpeedStrategyConfiguration, HighSpeedStrategyConfiguration
        ],
    ) -> List[FrenetPath]:

        if s0 >= strategy_config.stop_distance_m:
            # is this necessary?
            return []

        fplist = []
        for s in np.arange(
            strategy_config.stop_distance_m - strategy_config.stop_sampling_m,
            strategy_config.stop_distance_m + strategy_config.stop_sampling_m,
            strategy_config.stop_sampling_m,
        ):
            fp = FrenetPath()
            lon_qp = QuinticPolynomial(s0, s_d0, s_dd0, s, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, sampling_time)]
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            fplist.append(fp)
        return fplist

    def calc_destination_cost(
        self,
        target_speed: float,
        fp: FrenetPath,
        strategy_config: Union[
            LowSpeedStrategyConfiguration, HighSpeedStrategyConfiguration
        ],
    ):
        ds = (strategy_config.stop_distance_m - fp.s[-1]) ** 2
        return CostWeight.K_S * ds
