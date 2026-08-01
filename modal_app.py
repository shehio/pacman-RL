"""Run DQN training for Ms. Pac-Man on Modal GPUs.

Usage (requires `pip install modal`, `modal token new`, and a Modal secret named
"wandb" containing WANDB_API_KEY):

    modal run modal_app.py                                  # defaults: 2M steps, vanilla targets
    modal run modal_app.py --lr 0.00025 --double-dqn        # Double DQN A/B arm
    modal run modal_app.py --max-steps 500000 --update-freq 1000

Training logs stream to wandb (entity "shehio", project "pacman-rl"), so the
ephemeral container filesystem is fine; pass --no-wandb for a throwaway run that
writes nothing anywhere.
"""

import modal

app = modal.App("pacman-rl-dqn")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.2",
        "gymnasium[atari]>=0.29",
        "ale-py>=0.9",
        "scikit-image>=0.22",
        "numpy",
        "wandb",
    )
    .add_local_dir(
        "pacman_full_DQN",
        remote_path="/root/pacman_rl",
        ignore=["saved_model", "logs", "*.csv", "__pycache__"],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 8,
    secrets=[modal.Secret.from_name("wandb")],
)
def train(
    lr: float = 0.005,
    max_steps: int = 2_000_000,
    update_freq: int = 10_000,
    learn_start: int = 10_000,
    double_dqn: bool = False,
    wandb_tags: str = "modal",
    no_wandb: bool = False,
):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "/root/pacman_rl/pacman_agent_torch.py",
        "--lr", str(lr),
        "--max-steps", str(max_steps),
        "--update-freq", str(update_freq),
        "--learn-start", str(learn_start),
        "--wandb-tags", wandb_tags,
        "--no-watch",
        "--no-load",
        "--no-save",
    ]
    if double_dqn:
        cmd.append("--double-dqn")
    if no_wandb:
        cmd.append("--no-wandb")
    subprocess.run(cmd, check=True, cwd="/root/pacman_rl")


@app.local_entrypoint()
def main(
    lr: float = 0.005,
    max_steps: int = 2_000_000,
    update_freq: int = 10_000,
    learn_start: int = 10_000,
    double_dqn: bool = False,
    wandb_tags: str = "modal",
    no_wandb: bool = False,
):
    train.remote(
        lr=lr,
        max_steps=max_steps,
        update_freq=update_freq,
        learn_start=learn_start,
        double_dqn=double_dqn,
        wandb_tags=wandb_tags,
        no_wandb=no_wandb,
    )
