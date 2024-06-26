#!/bin/sh

# Check if conda is installed
if command -v conda &> /dev/null
then
    current_env=$(conda info --envs | grep '*' | awk '{print $1}')
    num_packages=$(conda list | grep -v "^#" | wc -l)

    echo current_env $current_env
    echo num_packages $num_packages


    if [ "$num_packages" -eq 0 ]; then
        conda env update --name $current_env --file environment.yml --prune
    
    else
        echo "The environment '$current_env' is not empty. Please run shell script in empty conda enviornment."
        exit 1
        
    fi
else
    # Check if pip is installed
    if command -v pip &> /dev/null
    then
        echo "pip found. Installing packages with pip..."
        pip install matplotlib
        pip install numpy
        pip install scikit-learn
        pip install sklearn
        pip install tqdm
        pip install termcolor
    # Check if pip3 is installed
    elif command -v pip3 &> /dev/null
    then
        echo "pip3 found. Installing packages with pip3..."
        pip3 install matplotlib
        pip3 install numpy
        pip3 install scikit-learn
        pip3 install sklearn
        pip3 install tqdm
        pip3 install termcolor
    else
        echo "Neither conda, pip, nor pip3 found. Please install one of them first."
        exit 1
    fi
fi





