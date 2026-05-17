"""Default SB3-PPO trainer (the injected seam's default impl).

Lazy: stable-baselines3 + gymnasium are imported ONLY inside .train(), so
importing this module (or the pipeline) needs no RL deps. Local by
default; device auto-detected (CUDA→MPS→CPU). Tests inject their own
trainer and never reach this code.
"""
import numpy as np

from .device import device_str


def make_ppo_trainer(total_timesteps: int = 200_000, **ppo_kwargs):
    return _SB3Trainer(total_timesteps, ppo_kwargs)


class _EpisodeSamplingEnv:
    """Wrap a list of SingleTradeEnv episodes as one gymnasium Env: each
    reset() samples an episode (the agent learns one shared policy across
    all trades). Built lazily so gymnasium is only required at train time.
    """
    def __init__(self, episodes, seed):
        import gymnasium as gym
        self._g = gym
        self.eps = [e for _, e in episodes]
        self.rng = np.random.default_rng(seed)
        e0 = self.eps[0] if self.eps else None
        obs_dim = e0.obs_dim if e0 else 1
        # shared, evolving account state (same dict the episode envs hold,
        # so augment_obs sees it); strategy.shape_reward shapes the terminal
        # reward and may StopIteration = account blown → reset the simulated
        # account so PPO experiences many lifetimes and learns to avoid zero.
        self.strategy = getattr(e0, "strategy", None)
        self.run_state = getattr(e0, "run_state", {"cum_r": []})
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.cur = self.eps[int(self.rng.integers(len(self.eps)))]
        return self.cur.reset(), {}

    def step(self, action):
        obs, r, term, trunc, info = self.cur.step(action)
        if (term or trunc) and self.strategy is not None:
            try:
                r = float(self.strategy.shape_reward(r, self.run_state))
                self.run_state["cum_r"].append(r)
            except StopIteration:               # account blown
                r = -1.0                        # strong terminal penalty
                self.run_state["cum_r"].clear()  # reset simulated account
        return obs, r, term, trunc, info


class _SB3Trainer:
    def __init__(self, total_timesteps, ppo_kwargs):
        self.total_timesteps = total_timesteps
        self.ppo_kwargs = ppo_kwargs

    def train(self, episodes, seed):
        if not episodes:
            return lambda obs: 0                       # nothing to learn
        try:
            import gymnasium  # noqa: F401
            from stable_baselines3 import PPO
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "RL training requires stable-baselines3 + gymnasium "
                "(pip install stable-baselines3 gymnasium). The pipeline "
                "itself is dep-free; only the default PPO trainer needs them."
            ) from e
        env = _EpisodeSamplingEnv(episodes, seed)
        model = PPO("MlpPolicy", env, seed=seed,
                    device=device_str("auto"), verbose=0, **self.ppo_kwargs)
        model.learn(total_timesteps=self.total_timesteps)

        def policy(obs):
            a, _ = model.predict(np.asarray(obs, np.float32),
                                 deterministic=True)
            return int(a)
        return policy
