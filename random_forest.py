'''
An example for random forest regression on Life Expectancy
'''

from utils import *
from sklearn.model_selection import train_test_split

filepath = 'data/Life_Expectancy_Data_processed .csv'
test_size = 0.2

df = load_dataset(filepath)
df = df.iloc[:, 1:]  # Remove Country name

# Prepare training set and testing set

features = df.copy()
labels = features.pop('Life expectancy')

# Remove features as desired
features.drop(['Infant Mort', 'Adult Mort'], axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, random_state=0, test_size=test_size
)

transformer = build_transformer()
regressor = build_regressor()

model = train_model(
    transformer, regressor, train_features, train_labels
)

predictions = predict(model, test_features)

plot_results(test_labels, predictions, 'outputs/RF')