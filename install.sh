#!/bin/sh

packages_for_installation="matplotlib numpy scikit-learn sklearn install tqdm termcolor"

verify_package_installation(){
    packages=$1
    missing_packages=""

    for package in packages; do
        if ! conda list | grep package; then
            missing_packages = "$missing_packages package"
        fi 
    done

    if [ -n "$missing_packages" ]; then
        echo "WARNING: The following packages failed to install: $missing_packages"
        exit 1
    else
        echo "Succesfully installed all packages"
        exit 0
    fi
}


# Check if conda is installed
if command -v conda &> /dev/null
then
    current_env=$(conda info --envs | grep '*' | awk '{print $1}')
    num_packages=$(conda list | grep -v "^#" | wc -l)


    if [ "$num_packages" -eq 0 ]; then
        conda env update --name $current_env --file environment.yml --prune
        verify_package_installation $packages_for_installation

    else
        echo "The environment '$current_env' is not empty. Please run shell script in empty conda enviornment."
        exit 1
        
    fi
else
    # Check if pip is installed
    if command -v pip &> /dev/null
    then
        echo "pip found. Installing packages with pip..."
        pip install $packages_for_installation
        verify_package_installation $packages_for_installation

    # Check if pip3 is installed
    elif command -v pip3 &> /dev/null
    then
        echo "pip3 found. Installing packages with pip3..."
        pip3 install $packages_for_installation
        verify_package_installation $packages_for_installation

    else
        echo "Neither conda, pip, nor pip3 found. Please install one of them first (preferbly conda)."
        exit 1
    fi
fi





