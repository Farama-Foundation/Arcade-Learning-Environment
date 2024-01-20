# ALE-docs

This folder contains the documentation for [Arcade-Learning-Environment](https://github.com/Farama-Foundation/arcade-Learning-Environment).

## Editing an environment page

If you are editing an Atari environment, directly edit the Markdown file in this repository.

Otherwise, fork Gymnasium and edit the docstring in the environment's Python file. Then, pip install your Gymnasium fork and run `docs/_scripts/gen_mds.py` in this repo. This will automatically generate a Markdown documentation file for the environment.

## Build the Documentation

Install the required packages and Gymnasium (or your fork):

```
pip install -r docs/requirements.txt
```

To build the documentation once:

```
cd docs
make html
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b html . _build
```
