from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import lancedb

# turn lanceDB into a numpy array, X (features) (one for each model) and Y (target)
# train multi output classifiers using logistic regression
# address class imbalance with balanced weights
# la fin