ml cuDNN/8.4.1.50-CUDA-11.6.0

conda create -n blendface python=3.8 -y
conda activate blendface
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install gdown
pip install -e ../data
pip install insightface
pip install face-alignment
pip install onnxruntime-gpu==1.13.1

mkdir -p checkpoints
cd checkpoints
gdown "https://drive.google.com/uc?id=1wFkGXI36lZZQpOeIuM_0BxX2rIYSIA1K" 
gdown "https://drive.google.com/uc?id=1FSCUC5CbyPKnl5Bbt58tPcKCVOyyt004" 
gdown https://drive.google.com/uc?id=1ssTKnNVGomtrtPl57EOGzKNz62bh6mSs
cd -