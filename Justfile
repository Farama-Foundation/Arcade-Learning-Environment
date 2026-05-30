set shell := ["bash", "-cu"]

venv := ".venv-docs"
python := venv + "/bin/python"

default:
    @just --list

docs-install:
    python3 -m venv {{venv}}
    {{python}} -m pip install --upgrade pip
    {{python}} -m pip install ale-py
    {{python}} -m pip install -r docs/requirements.txt

docs-serve: docs-install
    cd docs && ../{{venv}}/bin/sphinx-autobuild -b html . _build --open-browser

docs-build: docs-install
    cd docs && ../{{venv}}/bin/sphinx-build -b html . _build/html

docs-clean:
    rm -rf docs/_build docs/jupyter_execute
