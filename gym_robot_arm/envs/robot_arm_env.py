"""
Gymnasium 2D Robot Arm Environment.

A two-link robot arm reaching a randomly generated target point.
Built on PyGame rendering. Compatible with RL frameworks via Gymnasium API.
"""

import math
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_z(theta: float) -> np.ndarray:
    """Homogeneous Z-axis rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float32)


def _translation(dx: float, dy: float) -> np.ndarray:
    """Homogeneous 2D translation matrix (in XY plane)."""
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float32)


def _forward_kinematics(theta: np.ndarray, links: list[float]) -> list[np.ndarray]:
    """Compute homogeneous transforms for each joint + tip."""
    frames = [np.eye(4, dtype=np.float32)]
    for i in range(len(links)):
        frames.append(
            frames[-1] @ _rotation_z(theta[i]) @ _translation(links[i], 0.0)
        )
    return frames


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = (600, 600)
LINK_LENGTHS = [100.0, 100.0]
N_LINKS = len(LINK_LENGTHS)
MAX_REACH = sum(LINK_LENGTHS)

SCREEN_COLOR = (50, 168, 52)
LINK_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 0, 0)
TIP_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 0, 0)

FPS = 60

# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------

class RobotArmEnv(gym.Env):
    """
    Base class for 2-link robot arm reaching.

    Observation (4,): [target_x, target_y, joint_0, joint_1]
    """
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.links = LINK_LENGTHS
        self.n_links = N_LINKS
        self.max_reach = MAX_REACH
        self.centre = [WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2]

        # Angle bounds (0°–90°)
        self.min_theta = 0.0
        self.max_theta = math.radians(90)

        # Observation: target_x, target_y, joint_0, joint_1
        high = np.array([self.max_reach, self.max_reach, self.max_theta, self.max_theta],
                        dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Rendering
        self.render_mode = render_mode
        self._surface: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._pygame_inited = False

    # -- seeding -----------------------------------------------------------

    def seed(self, seed: Optional[int] = None):
        """Legacy compatibility seed."""
        np.random.seed(seed)

    # -- helpers -----------------------------------------------------------

    def _generate_random_angle(self) -> np.ndarray:
        return np.random.uniform(self.min_theta, self.max_theta, size=(self.n_links,))

    def _generate_random_pos(self) -> np.ndarray:
        """Generate a reachable target position."""
        theta = self._generate_random_angle()
        frames = _forward_kinematics(theta, self.links)
        return np.array([frames[-1][0, 3], frames[-1][1, 3]], dtype=np.float32)

    # -- step/reset -------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.theta = np.zeros(self.n_links, dtype=np.float32)
        self.target_pos = self._generate_random_pos()
        obs = np.concatenate([self.target_pos, self.theta]).astype(np.float32)
        info = {"target": self.target_pos.copy()}
        return obs, info

    def step(self, action):
        raise NotImplementedError

    # -- rendering --------------------------------------------------------

    def _init_pygame(self):
        if not self._pygame_inited:
            pygame.init()
            self._pygame_inited = True
        self._surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("RobotArm-Env")
        self._clock = pygame.time.Clock()

    def _draw_arm(self, theta: np.ndarray):
        """Draw the robot arm on the surface."""
        frames = _forward_kinematics(theta, self.links)
        origin_to_base = _translation(self.centre[0], self.centre[1])
        base = np.eye(4, dtype=np.float32) @ origin_to_base

        prev = base.copy()
        tip_pos = None
        for i in range(1, len(frames)):
            cur = base @ frames[i]
            p1 = (int(prev[0, 3]), int(prev[1, 3]))
            p2 = (int(cur[0, 3]), int(cur[1, 3]))
            pygame.draw.line(self._surface, LINK_COLOR, p1, p2, 5)
            pygame.draw.circle(self._surface, JOINT_COLOR, p1, 10)
            prev = cur.copy()
            tip_pos = p2

        if tip_pos:
            pygame.draw.circle(self._surface, TIP_COLOR, tip_pos, 8)

    def _draw_target(self):
        """Draw the target point."""
        base = _translation(self.centre[0], self.centre[1])
        target_world = base @ _translation(self.target_pos[0], -self.target_pos[1])
        pos = (int(target_world[0, 3]), int(target_world[1, 3]))
        pygame.draw.circle(self._surface, TARGET_COLOR, pos, 12)

    def render(self):
        if self.render_mode != "human":
            return
        if self._surface is None:
            self._init_pygame()

        self._surface.fill(SCREEN_COLOR)
        self._draw_target()
        self._draw_arm(self.theta)
        self._clock.tick(FPS)
        pygame.display.flip()

    def close(self):
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False
        self._surface = None
        self._clock = None


# ---------------------------------------------------------------------------
# V0 — Discrete actions
# ---------------------------------------------------------------------------

class RobotArmEnvV0(RobotArmEnv):
    """
    V0: Discrete 7-action space.

    Each action increments/decrements joint angles by a fixed rate.
    Reward: +1 when tip is within 10 px of target, -1 when farther than previous step.
    Episode ends after accumulated reward reaches ±10.
    """

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        self.inc_rate = math.radians(1)  # approx 1° per step

        # 7 discrete actions
        self._actions = {
            0: (0.0, 0.0, "HOLD"),
            1: (self.inc_rate, 0.0, "INC_J1"),
            2: (-self.inc_rate, 0.0, "DEC_J1"),
            3: (0.0, self.inc_rate, "INC_J2"),
            4: (0.0, -self.inc_rate, "DEC_J2"),
            5: (self.inc_rate, self.inc_rate, "INC_J1_J2"),
            6: (-self.inc_rate, -self.inc_rate, "DEC_J1_J2"),
        }
        self.action_space = spaces.Discrete(len(self._actions))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_error = math.inf
        self.score = 0
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        d0, d1, _ = self._actions[action]
        self.theta[0] = float(np.clip(self.theta[0] + d0, self.min_theta, self.max_theta))
        self.theta[1] = float(np.clip(self.theta[1] + d1, self.min_theta, self.max_theta))

        # Compute tip position
        frames = _forward_kinematics(self.theta, self.links)
        tip_pos = np.array([frames[-1][0, 3], frames[-1][1, 3]], dtype=np.float32)
        error = _distance(self.target_pos, tip_pos)

        # Reward
        reward = 0.0
        if error >= self.current_error:
            reward = -1.0
        epsilon = 10.0
        if abs(error) < epsilon:
            reward = 1.0

        self.current_error = error
        self.score += reward

        terminated = abs(self.score) >= 10.0
        truncated = False

        obs = np.concatenate([self.target_pos, self.theta]).astype(np.float32)
        info = {
            "distance_error": error,
            "target": self.target_pos.copy(),
            "tip": tip_pos,
        }
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# V1 — Continuous actions (position control)
# ---------------------------------------------------------------------------

class RobotArmEnvV1(RobotArmEnv):
    """
    V1: Continuous 2D action space.

    Action values in [-1, 1] are linearly mapped to joint angle range [0°, 90°].
    Reward: negative normalised distance to target.
    Episode terminates when tip is within 5 px of target.
    """

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Map [-1, 1] → [min_theta, max_theta]
        self.theta[0] = float(np.interp(action[0], [-1.0, 1.0], [self.min_theta, self.max_theta]))
        self.theta[1] = float(np.interp(action[1], [-1.0, 1.0], [self.min_theta, self.max_theta]))

        frames = _forward_kinematics(self.theta, self.links)
        tip_pos = np.array([frames[-1][0, 3], frames[-1][1, 3]], dtype=np.float32)
        error = _distance(self.target_pos, tip_pos)

        reward = -error / float(self.max_reach)
        terminated = error < 5.0
        truncated = False

        obs = np.concatenate([self.target_pos, self.theta]).astype(np.float32)
        info = {
            "distance_error": error,
            "target": self.target_pos.copy(),
            "tip": tip_pos,
        }
        return obs, reward, terminated, truncated, info
