# README

Add badges - https://github.com/dwyl/repo-badges

- status: stable
- license: ...
- codecov: ...
- code style: ... (black)


## Tools

### Documentation

Build (w/i docs directory)

```commandline
make html
```

Clean (w/i docs directory)

```commandline
make clean
```

### Run unittests

```commandline
python -m unittest discover tests
```

### Run coverage report

```commandline
coverage run -m unittest discover tests
coverage html --omit="*/test*"
```

### Run black autoformatter

```commandline
python -m black torch_fairness
```

### Hide cell in notebook

Edit cell metadat and add "nbsphinx": "hidden"