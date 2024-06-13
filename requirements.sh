pip install torch torchvision
conda install -y numpy scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom freetype-py shapely
pip install opencv-python==4.5.4.60  
pip install kornia==0.6.8
pip install wandb
pip install shapely vharfbuzz==0.2.0
pip install diffusers==0.8
pip install easydict matplotlib 
pip install transformers scipy ftfy accelerate filelock tqdm huggingface-hub 
pip install aspose-words
pip install pip install svg.path
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
DIFFVG_CUDA=1 python setup.py install
pip 