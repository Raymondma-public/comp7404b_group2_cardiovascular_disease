import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from os.path import join

np.set_printoptions(precision=3)
pd.set_option('precision', 3)

def load_dataset(filepath):
    return pd.read_csv(
        filepath, header=0, na_values='?',
        comment='\t', sep=',', skipinitialspace=True
    )

def build_transformer(weights='uniform', k=5, metric='nan_euclidean'):
    return FeatureUnion(
        transformer_list=[
            ('features', KNNImputer(
                missing_values=np.nan,
                weights=weights,
                metric=metric,
                n_neighbors=k)),
        ]
    )

def build_regressor(n_tree=100, criterion='mse', max_depth=20, random_state=0):
    return RandomForestRegressor(
        n_estimators=n_tree,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    )

def train_model(transformer, regressor, train_features, train_labels):
    pipeline = make_pipeline(transformer, regressor)
    trained_model = pipeline.fit(train_features, train_labels)
    return trained_model

def predict(trained_model, features):
    return trained_model.predict(features)


def plot_results(labels, preds, outpath):

    # Actual versus expected
    plt.clf()
    ave, ax = plt.subplots()
    ax.set_title('Actual vs Fitted Life Expectancy')
    ax.set_xlabel('Actual (Years)')
    ax.set_ylabel('Fitted (Years)')
    ax.scatter(labels, preds)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.savefig(join(outpath, 'ave.png'))

    plt.clf()
    err, ax2 = plt.subplots()
    ax2.set_title('Error of Prediction')
    ax2.set_xlabel('Error (years)')
    ax2.set_ylabel('Density')
    ax2.hist(preds - labels, bins=50, density=True)
    plt.savefig(join(outpath, 'error_hist.png'))
