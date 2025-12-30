# ImMimic-CoRL2025


## Setup

1. Create the conda env
```
conda create -n immimic python=3.10
conda activate immimic
python -m pip install -U pip setuptools wheel
```

2. Install MuJoCo
```
pip install "mujoco==3.3.0"
```

3. Install PyTorch
```
pip install torch==2.6.0 torchvision==0.21.0
```

4. Install robosuite v1.5.1
```
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout v1.5.1 
pip install -e . --no-deps
cd ..
```
5. Install requirements.txt
```
pip install -r requirements.txt
```
