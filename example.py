from data_loader import DataLoader
from model_trainer import ModelTrainer
from models import *


DATA_FOLDER = 'data'

for smell_class in ['data-class', 'feature-envy', 'god-class', 'long-method']:
    data_loader = DataLoader(DATA_FOLDER, smell_class + '.arff')
    X_train, X_test, y_train, y_test = data_loader.get_data(test_ratio=0.15)

    # Example of using grid search to find best parameters set

    # model = DecisionTree(grid_search=True)

    # depths = np.arange(5, 21)
    # num_leafs = [1, 5, 10, 20, 50, 100]
    # param_grid = {'criterion': ['gini', 'entropy'],
    #               'max_depth': depths, 'min_samples_leaf': num_leafs}
    # model.build_grid_search_model(
    #     param_grid, cv=10, scoring='accuracy', return_train_score=True)

    # Example of normal training

    model = RandomForest()

    model_trainer = ModelTrainer(model, X_train, y_train, X_test, y_test)
    model_trainer.train()
    stats = model_trainer.get_statistics()

    print(f'Statistics for smell type {smell_class}:\n'
          f'\t{stats}\n')
