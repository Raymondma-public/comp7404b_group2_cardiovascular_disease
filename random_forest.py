'''
An example for random forest regression on Life Expectancy
'''

from utils import *
from sklearn.model_selection import train_test_split

filepath = 'data/Life_Expectancy_Data_processed .csv'
model_path= 'models/RF/model.joblib'
test_size = 0.2

df = load_dataset(filepath)
df = df.iloc[:, 1:]  # Remove Country name

# Prepare training set and testing set

features = df.copy()
labels = features.pop('Life expectancy')

# Remove features as desired
features = features.drop(['Infant Mort', 'Adult Mort'], axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, random_state=0, test_size=test_size
)

transformer = build_transformer()
regressor = build_regressor()

model = train_model(
    transformer, regressor, train_features, train_labels
)

# Save model
print()
print("=========Save model==========")
save_model(model, model_path)
predictions = predict(model, test_features)
train_score=score(model,train_features,train_labels)
test_score=score(model,test_features,test_labels)
print(f'Train r2 score:{train_score}\nTest r2 score:{test_score}')

print()
print("=========Save plot==========")
plot_results(test_labels, predictions, 'outputs/RF')

# Load model
print()
print("=========Load model==========")
loaded_model = load_model(model_path)
predictions = predict(loaded_model, test_features)
train_score=score(model,train_features,train_labels)
test_score=score(model,test_features,test_labels)
print(f'Train r2 score: {train_score}\nTest r2 score: {test_score}')

print()
print("=========Feature importance==========")
table = rf_features_importance(train_features.columns, model)
importance=pd.DataFrame(table, columns=["Feature", "Importance(%)"])
print(importance)
