"""Self-play callback utilities used by the trainer."""

from stable_baselines3.common.callbacks import BaseCallback


class SimpleSelfPlayCallback(BaseCallback):
    """Inject model references into envs and log self-play experience counts."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.multi_player_episodes = 0
        self.total_experiences_collected = 0
        self.model_injected = False

    def _on_training_start(self) -> None:
        # Handle VecEnv wrappers used by trainer.
        if hasattr(self.training_env, "envs"):
            for env in self.training_env.envs:
                target_env = self._get_base_env(env)
                if hasattr(target_env, "set_model_reference"):
                    target_env.set_model_reference(self.model)
                    self.model_injected = True
        elif hasattr(self.training_env, "env_method"):
            try:
                self.training_env.env_method("set_model_reference", self.model)
                self.model_injected = True
            except Exception:
                pass
        else:
            target_env = self._get_base_env(self.training_env)
            if hasattr(target_env, "set_model_reference"):
                target_env.set_model_reference(self.model)
                self.model_injected = True

    def _get_base_env(self, env):
        if hasattr(env, "env") and hasattr(env.env, "set_model_reference"):
            return env.env
        if hasattr(env, "set_model_reference"):
            return env
        if hasattr(env, "unwrap"):
            unwrapped = env.unwrap()
            if hasattr(unwrapped, "set_model_reference"):
                return unwrapped
        return env

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if info.get("episode_complete", False) and "multi_player_experiences" in info:
                    experiences = info["multi_player_experiences"]
                    self.multi_player_episodes += 1
                    self.total_experiences_collected += len(experiences)

                    if self.verbose >= 1 and self.multi_player_episodes % 10 == 0:
                        avg_exp = self.total_experiences_collected / self.multi_player_episodes
                        self.logger.record("self_play/episodes", self.multi_player_episodes)
                        self.logger.record("self_play/total_experiences", self.total_experiences_collected)
                        self.logger.record("self_play/avg_exp_per_episode", avg_exp)
                        self.logger.record("self_play/model_injected", float(self.model_injected))

        return True
