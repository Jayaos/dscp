"""Load experiment configs with a timestamp that stays fixed within each run."""

from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now().strftime(pattern), use_cache=True,
    )


def load_experiment_config(path):
    """Load a fresh config; cached ${now:...} values belong to this config only."""
    return OmegaConf.load(path)


def resolve_job_saving_dir(config, output_root):
    """Override results/<method> while retaining the configured run subfolders."""
    configured_path = Path(str(config.saving_dir))
    if len(configured_path.parts) < 3 or configured_path.parts[0] != "results":
        raise ValueError("Job saving_dir must start with results/<method>/.")
    output_dir = Path(output_root).expanduser().resolve().joinpath(*configured_path.parts[2:])
    config.saving_dir = str(output_dir)
    return output_dir
