import math
import pandas as pd
from scipy.stats import entropy
        
class NodeDecisionTree():
    def __init__(self, features, target, depth=1, value_best_split='', max_depth=3, min_n_obs=5):
        self.depth = depth
        self.features = features
        self.target = target
        self.children = []
        self.best_feature = None
        self.value_best_split = value_best_split
        self.max_depth = max_depth
        self.min_n_obs = min_n_obs
        self.entropy = self.entropy_score(self.target)

    def entropy_score(self, t):
        """Calculate the entropy score for a given target variable t.

        Args:
            t (pd.Series): A series containing values of a categorical variable.

        Returns:
            float: The entropy score calculated as -sum(p*log2(p)) where p is 
            the proportion of each unique value in t.

        """
        class_distribution = t.value_counts(normalize=True).tolist()
        sum_entropy = 0
        for c in class_distribution:
            sum_entropy += (c*math.log(c,2))

        return -sum_entropy
    
    def best_feature_to_split(self):
        """Find the best feature to split the data at this node based on information gain.

        Information gain is defined as IG(S,A) = H(S) - sum(v/S)*(H(Sv)) where S is 
        a set of examples, A is an attribute or feature,
        v is a possible value for A,
        Sv is a subset of S with A=v,
        H(S) is entropy score for S,
        H(Sv) is entropy score for Sv

        Returns:
            str: The name of the feature that has maximum information gain among all features.

        """
        results = []
        for c in self.features.columns:
            entropy_after_split = 0
            for value, ratio in self.features[c].value_counts(normalize=True).to_dict().items():
                entropy_after_split += self.entropy_score(self.target.loc[self.features[c]==value]) * ratio
            results.append({
                'information_gain': self.entropy - entropy_after_split,
                'feature': c,
            })
        results = pd.DataFrame(results)
        print (results)
        return results.set_index('feature')['information_gain'].idxmax()

    
    def split(self): 
        """Split the data at this node into child nodes based on the best feature and value.

        This method recursively calls itself on the child nodes until one of the stopping criteria is met.

        """
        # Features with a single static value are information less, we filter them out
        self.features = self.features.loc[:,self.features.nunique()>1]
        
        # When to stop expanding the tree? There are several cases:
        if self.target.shape[0]<self.min_n_obs:
            print(0)
            return
        
        if self.target.nunique()<=1:
            print(1)
            return
            
        if self.features.shape[1]==0:
            print(2)
            return
        
        if self.depth>self.max_depth:
            print(3)
            return
        
        # Not more stopping criterion, let's select the feature we will use to expand the tree        
        self.best_feature = self.best_feature_to_split()
        print("best feature:", self.best_feature)
        
        # Create child for each unique values of the best feature
        for c in self.features[self.best_feature].unique():
            index = self.features[self.best_feature]==c
            child = NodeDecisionTree(self.features.loc[index,:], self.target.loc[index], self.depth+1, c)
            child.split()
            self.children.append(child)


import pandas as pd
import pickle
df = pd.read_pickle('titanic.pickle')
features = df.drop('alive', axis=1)
target = df['alive']
root = NodeDecisionTree(features, target) 
root.split()   


# Random Forest Molel fitting

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Convert categorical variables to numerical using one-hot encoding
features_categorical = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(features_categorical, target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=2000, random_state=42)

rf.fit(X_train, y_train)

# Random Forest Model Evaluation
y_pred = rf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.3f}")