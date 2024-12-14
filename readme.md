# Sandworm Detection Using Bayesian Inference

This project implements a probabilistic model to predict sandworm detection based on vibration amplitude and distance, inspired by the science fiction universe of *Dune*. Using conditional probabilities and Gaussian distributions, the model predicts whether sandworms detect a rhythmic vibration.

## Overview

Given a dataset of rhythmic vibrations with features `Amplitude` and `Distance`, this model calculates:
- `P(Detect | a, d)`: The probability that a sandworm detects vibrations at a given amplitude and distance.
- `P(No Detect | a, d)`: The probability that a sandworm does not detect vibrations.

The prediction is made based on whichever probability is higher.

## Approach

### Probabilistic Formula
Using Bayesian inference, we compute:
P(Detect | a, d) = P(a | Detect) * P(d | Detect) * P(Detect) / P(a, d)

Where:
- P(a | Detect) : Gaussian probability density for amplitude given detection.
- P(d | Detect) : Gaussian probability density for distance given detection.
- P(a, d) : Total probability of observing  a  and  d .

Similar logic is used for  P(No Detect | a, d) .

## Dataset
1. `detection_data.csv`: The primary dataset for training and validation.
2. `detection_data_extra.csv`: An additional dataset for testing the model's accuracy.

### Features
- `Amplitude`: The vibration amplitude.
- `Distance`: The distance to the nearest sandworm.
- `Detection`: The outcome (`Detect` or `No Detect`).

## Results
The model outputs the number of correct and incorrect predictions for both the training and extra datasets.

## Usage
Run the script as follows:
1. Place `detection_data.csv` and `detection_data_extra.csv` in the same directory as the script.
2. Install dependencies: `numpy`, `pandas`.
3. Execute the script to view prediction results.
