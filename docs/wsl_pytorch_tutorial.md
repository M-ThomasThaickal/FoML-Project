# Instructions to set up WSL2 and PyTorch

*Last Updated: 2/17/2026*

## Prerequisites

- Windows 11
- NVIDIA GPU (newer 10 series)
- Latest NVIDIA Windows driver (go to the NVIDIA driver website and install latest, usually no need to delete previous driver)
- Having Windows Terminal installed from the Microsoft Store will make things much easier

---

## 1. Install / Update WSL2

Open PowerShell as Administrator and run:

```powershell
wsl --install
wsl --update
wsl --set-default-version 2
```

> **Note:** `wsl --set-default-version 2` is not necessary — WSL2 is generally the default installation of WSL.

---

## 2. Install Ubuntu Linux Distro

Ubuntu is the best distro for NVIDIA CUDA Toolkit support.

```powershell
wsl --install -d Ubuntu
```

> This might require you to reboot your computer to finish installation. It might also say distro with same name is already installed, if so ignore this and move to the next step, your system already has ubuntu.

---

## 3. Verify GPU is Visible Inside WSL

1. In a new Windows Terminal window, run `wsl` to start WSL. This spins up a lightweight VM that runs a Linux Kernel on top of your Windows environment.
2. Run the following command:

```bash
nvidia-smi
```

This should return your NVIDIA driver version and the GPU the kernel can detect. This should match the GPU in your device.

> **Note:** WSL uses the drivers on your Windows level to interface with your GPU. Do **NOT** reinstall drivers in the WSL/Linux environment, as that can cause issues.

If this does not work, it likely means your NVIDIA drivers are not updated to the latest. Go to the NVIDIA website and download the latest driver for your device normally.

---

## 4. Install CUDA Toolkit Within WSL

> PyTorch comes with the needed parts of the CUDA toolkit to build without this, but installing it separately gives better functionality and makes debugging easier.

In your WSL environment, run:

```bash
sudo apt update
sudo apt install -y build-essential
```

Then follow the instructions in the CUDA WSL guide:
[https://docs.nvidia.com/cuda/wsl-user-guide/index.html](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

**Short rundown:**

1. Remove old GPG key
2. Install the Linux x86_64 CUDA Toolkit using the WSL-Ubuntu Package from:
   [https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

   Use these selections:
   - **Operating System:** Linux
   - **Architecture:** x86_64
   - **Distribution:** WSL-Ubuntu
   - **Version:** 2.0
   - **Installer Type:** deb (local)

3. This will return a set of instructions to run in your WSL instance.

---

## 5. Install PyTorch (Optional)

In Ubuntu, first create a virtual environment however you see fit, below is the code I use for venvs, involving creating a venvs folder outside the project directory:

```bash
python3 -m venv ~/venvs/torch
source ~/venvs/torch/bin/activate
pip install --upgrade pip
```

Then install PyTorch:

> **IMPORTANT:** Replace `cu124` with `cu126` in the command below to install a more updated version.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## 6. Verify PyTorch Can See Your GPU

Run the following to confirm everything is working:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

If CUDA is available and your GPU name is printed, the setup is complete!
