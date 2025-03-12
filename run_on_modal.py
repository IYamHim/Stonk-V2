import modal
import os
import subprocess
import sys

volume = modal.Volume.from_name("stonks-v2")
VOLUME_DIR = "/app"
GPU_USED = "A10"

SUPER_SAIYIN = "Super_Sayin_GRPO_Trainer_StageII"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        """ls && \
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
dpkg -i cuda-keyring_1.1-1_all.deb && \
apt-get update && \
apt-get install -y cuda-toolkit    
"""
    )
    .pip_install_from_requirements("requirements.txt", gpu=GPU_USED)
    .add_local_file("Stonk_Trainer.py", f"{VOLUME_DIR}/Stonk_Trainer.py")
    .add_local_dir(SUPER_SAIYIN, f"{VOLUME_DIR}/{SUPER_SAIYIN}")
)
app = modal.App("stonks-v2")


@app.function(image=image, volumes={VOLUME_DIR: volume}, gpu=GPU_USED)
async def run_stonk_trainer(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-5,
    kl_coef: float = 0.15,
    save_steps: int = 100,
    max_train_samples: int = 2000,
):
    os.chdir(VOLUME_DIR)
    model_path = os.path.join(VOLUME_DIR, "models")
    os.makedirs(model_path, exist_ok=True)
    """
    --train 
    --quantize 
    --epochs 5 
    --batch_size 8 
    --lr 1e-5 
    --kl_coef 0.15 
    --save_steps 100 
    --diverse_predictions 
    --max_train_samples 2000 
    """
    subprocess.run(
        [
            "python",
            "Stonk_Trainer.py",
            "--train",
            "--quantize",
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(lr),
            "--kl_coef",
            str(kl_coef),
            "--save_steps",
            str(save_steps),
            "--diverse_predictions",
            "--max_train_samples",
            str(max_train_samples),
            "--output_path",
            model_path,
        ]
    )


@app.function(image=image, volumes={VOLUME_DIR: volume}, gpu=GPU_USED)
async def run_stonk_tester():
    os.chdir(VOLUME_DIR)
    model_path = os.path.join(VOLUME_DIR, "models")
    os.makedirs(model_path, exist_ok=True)
    subprocess.run(
        [
            "python",
            "Stonk_Trainer.py",
            "--test",
            "--quantize",
            "--model_path",
            model_path,
        ]
    )


@app.function(image=image, volumes={VOLUME_DIR: volume}, gpu=GPU_USED)
async def run_super_saiyin_grpon(max_train_samples: int = 5000):
    os.chdir(os.path.join(VOLUME_DIR, SUPER_SAIYIN))
    model_path = os.path.join(VOLUME_DIR, "models")
    checkpoints = os.path.join(model_path, "checkpoints")
    best_model = os.path.join(checkpoints, "best_model")
    os.makedirs(model_path, exist_ok=True)
    subprocess.run(
        [
            "python",
            "grpo_stage2.py",
            "--stage1_model",
            best_model,
            "--quantize",
            "--natural_distribution",
            "--max_train_samples",
            str(max_train_samples),
            "--output_path",
            model_path,
        ]
    )
    
@app.local_entrypoint()
async def main():
    await run_stonk_trainer.remote.aio(epochs=5, batch_size=8, lr=1e-5, kl_coef=0.15, save_steps=100, max_train_samples=2000)
    await run_stonk_tester.remote.aio()
    await run_super_saiyin_grpon.remote.aio(max_train_samples=5000)
