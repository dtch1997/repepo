from repepo.variables import Environ
from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    project: str = field(default=Environ.WandbProject)
    entity: str = field(default=Environ.WandbEntity)
    name: str = "repepo"
    track: bool = field(default=False)


class Logger:
    def log(self, *args, **kwargs):
        raise NotImplementedError


class WandbLogger(Logger):
    def __init__(self, config: WandbConfig):
        self.config = config
        if self.config.track:
            import wandb

            self.wandb = wandb

    def __enter__(self):
        if self.config.track:
            self.wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.name,
            )
        return self

    def log(self, *args, **kwargs):
        if self.config.track:
            self.wandb.log(*args, **kwargs)
        # Else no-op

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.track:
            self.wandb.finish()
