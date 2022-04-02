# Repo structure

The functionality of all the files and folders are explained below.

```data/```

    This folder contains the four data files for training and evaluating models,
    each of which corresponds to one of the four code smell types used.

```data_loader.py```

    A python module that defines `DataLoader` class, which is used for loading
    code smell data from 'data/' folder. This class does more than reading
    arff files, but also some preprocessing work including data cleaning, data
    imputation, separating features from labels and splitting into training and
    test set.

```models.py```

    A python module that defines the four models, `DecisionTree`, `RandomForest`,
    `GaussianNaiveBayes` and `SupportVectorMachine`. Note that these four models
    all inherit from a base class `GridSearchBaseModel`, which is an abstract
    class that integrates the grid search algorithm into the model training
    process. The use of grid search algorithm is controlled by the argument
    `grid_search` when defining the model.
    Other than the integration of grid search algorithm, these four models are 
    simply wrapper models of corresponding scikit-learn models.

```model_trainer.py```

    A python module that defines `ModelTrainer` class, which wraps the whole
    process of training model and reporting statistics.

```example.py```

    Examples to use the `DataLoader`, `ModelTrainer` and all types of models.

```requirements.txt```

    The file that specifies the python packages used in this project.

# Contributing

## Python formatter

We use `autopep8` to format our code. Properly set your IDE to automatically 
format the python file before saving.

## Commit message convention

The commit message should be structured as follows, quoted from [here](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

The types are specified as followed:
- **fix**: patches a bug in the codebase.
- **feat**: introduces a new feature to the codebase.
- **style**: codebase changes related to code format problems
- **docs**
- **chore**
- **refactor**
- ......

## Commit as often as possible

Break down the code you implemented into small parts and commit as soon as you
finish a small part.

**DO NOT** squash all the work into one commit!