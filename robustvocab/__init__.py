"""RobustVocab — hint-free recovery and prompt-robust deployment (submission-b)."""

__all__ = ["run_robustvocab_episode"]


def __getattr__(name: str):
    if name == "run_robustvocab_episode":
        from robustvocab.pipeline import run_robustvocab_episode

        return run_robustvocab_episode
    raise AttributeError(name)
