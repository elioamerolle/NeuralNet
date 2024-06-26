#!/bin/sh

if command -v conda &> /dev/null
then
    
    conda install matplotlib==3.0.3
    conda install numpy==1.14.2
    conda install anaconda::scikit-learn
    conda install tqdm==4.64.1
    conda install termcolor==1.1.0
else

    echo "Conda not found. Installing packages with pip..."
    pip install matplotlib==3.0.3
    pip install numpy==1.14.2
    pip install scikit-learn==0.20.3
    pip install tqdm==4.64.1
    pip install termcolor==1.1.0

fi
