"""
Generate animated GIF demo of RobotArmEnv.
Shows the arm reaching a target using random actions.
"""

import os
import sys
from pathlib import Path

# Headless pygame — no display needed
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
from PIL import Image

# Register envs
import gym_robot_arm  # noqa: F401


def capture_frame(env, width=400, height=400) -> Image.Image:
    """Capture pygame surface as PIL Image."""
    import pygame
    env.render()
    # Access unwrapped env for internal attributes
    unwrapped = env.unwrapped
    surf = getattr(unwrapped, "_surface", None)
    if surf is None:
        # Force surface creation
        unwrapped._init_pygame()
        env.render()
        surf = unwrapped._surface
    raw = pygame.image.tostring(surf, "RGB")
    return Image.frombytes("RGB", surf.get_size(), raw).resize((width, height))


def generate_gif_v0(output_path: str, n_steps: int = 60):
    """V0 demo: discrete random actions."""
    env = gym.make("robot-arm-v0", render_mode="human")
    obs, _ = env.reset()

    frames = []
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        frames.append(capture_frame(env))
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    gif_path = str(Path(output_path) / "robot-arm-v0.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    print(f"V0 GIF saved: {gif_path}  ({len(frames)} frames)")
    return gif_path


def generate_gif_v1(output_path: str, n_steps: int = 20):
    """V1 demo: continuous actions — smooth reach."""
    env = gym.make("robot-arm-v1", render_mode="human")
    frames = []

    for ep in range(3):
        obs, _ = env.reset()
        for _ in range(n_steps):
            action = np.random.uniform(-0.3, 0.3, size=(2,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            frames.append(capture_frame(env))
            if terminated or truncated:
                break

    env.close()

    gif_path = str(Path(output_path) / "robot-arm-v1.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    print(f"V1 GIF saved: {gif_path}  ({len(frames)} frames)")
    return gif_path


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    os.makedirs(out_dir, exist_ok=True)
    generate_gif_v0(out_dir)
    generate_gif_v1(out_dir)
