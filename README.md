# Machine learning with kernel methods - Data challenge

## Introduction
This repository contains our work for the data challenge on Kaggle https://www.kaggle.com/c/advanced-learning-models-2020 proposed in the course Machine learning with kernel methods (MSIAM Ensimag).

The main goal of this challenge was to learn how to implement machine learning algorithms, gain understanding about them and adapt them to structural data. The task was to predict whether a DNA sequence region is binding site to a specific transcription factor. Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomic segments can be classified into two classes for a TF of interest: bound or unbound. During this challenge we worked on three datasets corresponding to the three TFs. 

In this challenge we could not use machine learning librairies.

## Structure
The repository contains the following folders:
* src: contains the scripts of all functions implemented during this project
* data: contains the csv files of the datasets (in raw or numerical form)
* gram_matrix: contains the gram matrices created during the fit of the methods
* predictions: contains the results of the prediction on the test set

The file src/start.py reproduces our best submission.

## Rank
**Score (public/private)**: $0.668$/$0.672$ 

**Rank (public/private)**: 18/20 over 32

## Author
**Clara Bourgoin**

## License
This project is licensed under the CC BY-NC-ND 4.0 License - see the [LICENSE.md](LICENSE.md) file for details

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
