"""System-specific configuration settings"""
from pathlib import Path

from yaml import safe_load
from pydantic import BaseModel, Field

AMBER_BIN_PATH = "/lus/eagle/projects/RL-fold/lward/multi-rl-workflow/env-amber/bin"
"""Path to the folder containing the ambertools binaries."""

PYMOL_PATH = "/lus/eagle/projects/RL-fold/lward/multi-rl-workflow/env-pymol/bin/pymol"
"""Path to the PyMol executable"""

WHAM_PATH = "/lus/eagle/projects/RL-fold/hengma/evb/wf_inputs/wham/wham/wham"
"""Path to the WHAM executable"""


class MultiRLConfig(BaseModel):
    """Configuration for a MultiRL run.

    These settings define behavior of the functions and how they are distributed across nodes.

    Paths to files should be either absolute or relative to the path of the YAML file given to :meth:`from_yaml`
    """

    # TODO (wardlt): Assert that paths are absolute

    # Settings related to RL training
    rl_retrain: bool = Field(..., description='Whether to retrain at all')
    rl_nodes: int = Field(..., description='Number of nodes per RL training')
    rl_num_folding_helpers: int = Field(..., description='Number of nodes devoted to computing folding scores for RL ranks')
    rl_initial_model_path: Path = Field(..., description='Path to the directory containing the initial model')
    rl_epochs: int = Field(..., description='Number of epochs per RL training task')
    rl_batch_size: int = Field(..., description='Number of sequences produced during each epoch')
    rl_mini_batch_size: int = Field(..., description='Number of sequences used per training set')
    rl_use_md_reward: bool = Field(..., description='Whether to use RMSF as one of the rewards')
    rl_reward_choice: str = Field(..., description='Which reward function to use')
    rl_prompt_diversity: bool = Field(..., description='Whether to use different prompts for each training point')

    # Settings related to RL rollout
    rollout_nodes: int = Field(..., description='Number of nodes used for rollout')
    rollout_episodes: int = Field(..., description='Number of episodes per RL rollout')
    rollout_batch_size: int = Field(..., description='Number of sequences to produce each rollout episode')

    # Settings related to the scoring
    scoring_batch_size: int = Field(..., description='Number of sequences to fold per task')

    # Settings related to MD
    md_yml_path: Path = Field(..., description='Path to the YAML file defining settings for MD')
    md_oversubscribe: int = Field(..., description='Number of MD tasks to place in queue in excess of the slots available')
    md_trailing_intolerance: int = Field(..., description='How many trailing tasks we will kill if they do not finish after the others')
    md_trailing_impatience: float = Field(..., description='How long to wait before killing trailing tasks in a batch')

    # Settings related to EVB
    evb_frequency: float = Field(..., description='How frequently to submit EVB computations.'
                                                  ' The ratio of MD to EVB is `evb_frequency`:`gpus_per_node * (1 - evb_frequency)`',
                                 ge=0, le=1)
    evb_archive_directory_path: Path = Field(..., description='Location in which to store the EVB results')
    evb_reference_pdb_path: Path = Field(..., description='Reference PDB to use for seeding EVB computation')
    evb_ligand_config_path: Path = Field(..., description='Path to the ligand configuration YAML file')
    evb_md_template_path: Path = Field(..., description='Path to the MD configuration template file')
    evb_tasks_per_gpu: int = Field(1, description='Number of MD tasks to run on each GPU')

    # Settings related to computational layout
    rollout_size: int | float = Field(..., description='Number of fraction of nodes devoted to rollout')
    gpus_per_node: int = Field(4, description='Number of GPUs per compute node')

    # Settings related to ablation tasks
    random_search: bool = Field(False, description='Disable ranking sequences by folding or MD scores')

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'MultiRLConfig':
        """Load in the configuration """
        with open(path) as fp:
            data: dict = safe_load(fp)

        # Ensure all paths are absolute
        yml_folder = Path(path).parent.absolute()
        for path_name in [f for f in data.keys() if f.endswith('_path')]:
            path = Path(data[path_name])
            if not path.is_absolute():
                path = yml_folder / path
            data[path_name] = path

        return cls(**data)
