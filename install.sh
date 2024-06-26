#!/bin/sh

#!/bin/sh

# Check if conda is installed
if command -v conda &> /dev/null
then
    echo "Conda found. Installing packages with conda..."
    conda install matplotlib -y
    conda install numpy==1.14.2 -y
    conda install anaconda::scikit-learn -y
    conda install tqdm==4.64.1 -y
    conda install termcolor==1.1.0 -y
else
    # Check if pip is installed
    if command -v pip &> /dev/null
    then
        echo "pip found. Installing packages with pip..."
        pip install matplotlib
        pip install numpy==1.14.2
        pip install scikit-learn==0.20.3
        pip install sklearn
	    pip install tqdm==4.64.1
        pip install termcolor==1.1.0
    # Check if pip3 is installed
    elif command -v pip3 &> /dev/null
    then
        echo "pip3 found. Installing packages with pip3..."
        pip3 install matplotlib
        pip3 install numpy==1.14.2
        pip3 install scikit-learn==0.20.3
        pip3 install sklearn
        pip3 install tqdm==4.64.1
        pip3 install termcolor==1.1.0
    else
        echo "Neither conda, pip, nor pip3 found. Please install one of them first."
        exit 1
    fi
fi





