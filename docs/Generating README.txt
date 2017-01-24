#!/bin/bash
pip install readme2tex

jupyter nbconvert --to markdown README_INPUT.ipynb 
python -m readme2tex --output README.md README_INPUT.md