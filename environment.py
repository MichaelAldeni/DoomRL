import gymnasium
from vizdoom import gymnasium_wrapper  # noqa


if __name__ == "__main__":
    env = gymnasium.make(
        "VizdoomBasic-v0", render_mode="human", frame_skip=4
    ) # VizdoomHealthGatheringSupreme-v0

    for _ in range(10):
        done = False
        obs, info = env.reset(seed=42)
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated