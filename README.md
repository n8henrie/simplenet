# SimpleNet

[![Build Status](https://travis-ci.org/n8henrie/simplenet.svg?branch=master)](https://travis-ci.org/n8henrie/simplenet)

A simple neural network in Python

- Free software: MIT
- Documentation: https://simplenet-nn.readthedocs.io

## Features

- Simple interface
- Minimal dependencies (numpy)
- Runs on Pythonista on iOS
- Attempts to verify accuracy by comparing results with popular frameworks
  Keras and Tensorflow

## Introduction

This is a simple multilayer perceptron that I decided to build as I learned a
little bit about machine learning and neural networks. It doesn't have many
features.

## Dependencies

- Python >= 3.5 (will likely require 3.6 eventually, if Pythonista updates)
- numpy

## Quickstart

1. `pip3 install simplenet`
1. See `examples/`

### Development Setup

1. Clone the repo: `git clone https://github.com/n8henrie/simplenet && cd
   simplenet`
1. Make a virtualenv: `python3 -m venv venv`
1. `source venv/bin/activate'
1. `pip install -e .[dev]`

## Acknowledgements

- Andrew Ng's Coursera courses

## TODO

I don't really know any Latex, so if anybody wants to help me fill out some of
the other docstrings with pretty equations, feel free. I'm also not a
mathematician, so if anything doesn't seem quite right, feel free to speak up.

## Troubleshooting / FAQ

- How can I install an older / specific version of SimpleNet?
    - Install from a tag:
        - pip install git+git://github.com/n8henrie/simplenet.git@v0.1.0
    - Install from a specific commit:
        - pip install git+git://github.com/n8henrie/simplenet.git@aabc123def456ghi789
