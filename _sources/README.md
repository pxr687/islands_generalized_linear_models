# Islands: Generalized Linear Models

An online textbook; a quick introduction to generalized linear models. Currently in first draft form.

## To Do

* Loop through pages and make sure everything is linked to 'dataframe' space, e.g. link the equations back to the example variables and show formulae embedded with data from the actual dataframes (e.g. within vectors/matrices)
* Tidy up and add comments to .py file.

### Jupyterbook build

jupyter-book build -W -n --keep-going LOCAL_PATH_TO_BOOK

ghp-import -n -p -f _build/html
