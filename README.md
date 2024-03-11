# MiHo

work in progress...

## Install

```bash
git clone https://github.com/fb82/MiHo.git
git checkout DIM-submodule
git submodule update --init --recursive
```

Create a new conda environment:

```
conda create -n miho python=3.9
conda activate miho
```

Install all the MiHo dependencies and pytorch:
```
conda install anaconda::pillow
conda install anaconda::numpy
conda install conda-forge::matplotlib

```

Install deep-image-matching as package:

```bash
cd ./thirdparty/deep-image-matching
pip install -e .
```
