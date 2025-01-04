import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.datasets import load_boston, load_iris, load_digits
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Index Page
def index_page():
    st.title("Machine Learning Algorithms Learning Resource")
    st.header("How to Select the Best Algorithm for Your Data")
    st.write("""
    Selecting the best machine learning algorithm for your data involves several considerations:

    1. **Type of Problem**:
       - **Supervised Learning**: Use for labeled data.
       - **Unsupervised Learning**: Use for unlabeled data.
       - **Reinforcement Learning**: Use for decision-making and control problems.
       - **Semi-Supervised Learning**: Use when you have a mix of labeled and unlabeled data.

    2. **Nature of Output**:
       - **Continuous Output**: Regression algorithms.
       - **Discrete Output**: Classification algorithms.

    3. **Data Characteristics**:
       - **Size of Data**: Some algorithms perform better with large datasets.
       - **Dimensionality**: High-dimensional data may require dimensionality reduction techniques.
       - **Noise**: Some algorithms are more robust to noise.

    4. **Performance Metrics**:
       - **Accuracy**: For classification problems.
       - **Mean Squared Error (MSE)**: For regression problems.
       - **Precision, Recall, F1-Score**: For imbalanced datasets.

    5. **Computational Resources**:
       - **Time Complexity**: Some algorithms are computationally intensive.
       - **Memory Requirements**: Consider the memory footprint of the algorithm.
    """)

# Supervised Learning Page
def supervised_learning_page():
    st.title("Supervised Learning Algorithms")
    st.header("Regression Algorithms")

    # Linear Regression
    st.subheader("Linear Regression")
    st.write("""
    **Explanation**:
    Linear Regression is a simple and commonly used algorithm for predicting a continuous output. It establishes a linear relationship between the input features and the output variable.

    **When to Use**:
    - Predicting house prices based on features like size, location, etc.
    - Forecasting sales based on advertising spend.
    """)
    st.code("""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston

    # Load dataset
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()
    """)

    # Polynomial Regression
    st.subheader("Polynomial Regression")
    st.write("""
    **Explanation**:
    Polynomial Regression is an extension of Linear Regression where the relationship between the input features and the output variable is modeled as an nth degree polynomial.

    **When to Use**:
    - When the relationship between features and the target is not linear.
    """)
    st.code("""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    # Load dataset
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()
    """)

    # Ridge Regression
    st.subheader("Ridge Regression")
    st.write("""
    **Explanation**:
    Ridge Regression is a type of Linear Regression that includes L2 regularization to prevent overfitting.

    **When to Use**:
    - When dealing with multicollinearity in the dataset.
    """)
    st.code("""
    from sklearn.linear_model import Ridge

    # Load dataset
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()
    """)

    # Lasso Regression
    st.subheader("Lasso Regression")
    st.write("""
    **Explanation**:
    Lasso Regression is a type of Linear Regression that includes L1 regularization to prevent overfitting and can perform feature selection.

    **When to Use**:
    - When you need to perform feature selection.
    """)
    st.code("""
    from sklearn.linear_model import Lasso

    # Load dataset
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()
    """)

    # ElasticNet Regression
    st.subheader("ElasticNet Regression")
    st.write("""
    **Explanation**:
    ElasticNet Regression combines both L1 and L2 regularization.

    **When to Use**:
    - When you need a balance between Ridge and Lasso Regression.
    """)
    st.code("""
    from sklearn.linear_model import ElasticNet

    # Load dataset
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()
    """)

    st.header("Classification Algorithms")

    # Logistic Regression
    st.subheader("Logistic Regression")
    st.write("""
    **Explanation**:
    Logistic Regression is used for binary classification problems. It models the probability of a binary outcome.

    **When to Use**:
    - Predicting whether an email is spam or not.
    - Predicting whether a customer will churn or not.
    """)
    st.code("""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.metrics import confusion_matrix, accuracy_score

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # Binary classification

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')

    # Plot
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """)

    # Decision Trees
    st.subheader("Decision Trees")
    st.write("""
    **Explanation**:
    Decision Trees are used for both classification and regression tasks. They split the data into subsets based on the value of input features.

    **When to Use**:
    - When you need interpretable models.
    - When dealing with non-linear relationships.
    """)
    st.code("""
    from sklearn.tree import DecisionTreeClassifier

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')

    # Plot
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """)

    # Random Forest
    st.subheader("Random Forest")
    st.write("""
    **Explanation**:
    Random Forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and control over-fitting.

    **When to Use**:
    - When you need a robust model with high accuracy.
    - When dealing with large datasets.
    """)
    st.code("""
    from sklearn.ensemble import RandomForestClassifier

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')

    # Plot
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """)

    # Naive Bayes
    st.subheader("Naive Bayes")
    st.write("""
    **Explanation**:
    Naive Bayes is a simple probabilistic classifier based on Bayes' theorem with strong independence assumptions.

    **When to Use**:
    - Text classification problems.
    - When you need a fast and simple model.
    """)
    st.code("""
    from sklearn.naive_bayes import GaussianNB

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')

    # Plot
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """)

    # SVM
    st.subheader("Support Vector Machines (SVM)")
    st.write("""
    **Explanation**:
    SVM is a powerful classifier that finds the hyperplane that best separates the classes.

    **When to Use**:
    - When you need a robust model with high accuracy.
    - When dealing with high-dimensional spaces.
    """)
    st.code("""
    from sklearn.svm import SVC

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')

    # Plot
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """)

    # KNN
    st.subheader("K-Nearest Neighbors (KNN)")
    st.write("""
    **Explanation**:
    KNN is a simple and intuitive classifier that classifies instances based on the majority vote of its k nearest neighbors.

    **When to Use**:
    - When you need a simple and interpretable model.
    - When dealing with small to medium-sized datasets.
    """)
    st.code("""
    from sklearn.neighbors import KNeighborsClassifier

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')

    # Plot
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """)

# Unsupervised Learning Page
def unsupervised_learning_page():
    st.title("Unsupervised Learning Algorithms")
    st.header("Clustering Algorithms")

    # K-Means Clustering
    st.subheader("K-Means Clustering")
    st.write("""
    **Explanation**:
    K-Means Clustering is a simple and popular unsupervised machine learning algorithm used for clustering data into K distinct groups.

    **When to Use**:
    - Customer segmentation.
    - Image compression.
    """)
    st.code("""
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Initialize and fit model
    model = KMeans(n_clusters=4)
    model.fit(X)

    # Predict
    y_pred = model.predict(X)

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X')
    plt.show()
    """)

    # DBSCAN
    st.subheader("DBSCAN")
    st.write("""
    **Explanation**:
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are packed closely together.

    **When to Use**:
    - When you need to find clusters of arbitrary shape.
    - When dealing with noise in the data.
    """)
    st.code("""
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Initialize and fit model
    model = DBSCAN(eps=0.3, min_samples=10)
    model.fit(X)

    # Predict
    y_pred = model.labels_

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.show()
    """)

    # Hierarchical Clustering
    st.subheader("Hierarchical Clustering")
    st.write("""
    **Explanation**:
    Hierarchical Clustering builds a hierarchy of clusters by recursively merging or dividing clusters.

    **When to Use**:
    - When you need a dendrogram to visualize the clustering process.
    - When dealing with small to medium-sized datasets.
    """)
    st.code("""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import make_blobs
    import scipy.cluster.hierarchy as sch

    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Initialize and fit model
    model = AgglomerativeClustering(n_clusters=4)
    model.fit(X)

    # Predict
    y_pred = model.labels_

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.show()

    # Dendrogram
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.show()
    """)

    st.header("Dimensionality Reduction Algorithms")

    # PCA
    st.subheader("Principal Component Analysis (PCA)")
    st.write("""
    **Explanation**:
    PCA is a dimensionality reduction technique that transforms the data into a new coordinate system where the greatest variances by any projection of the data come to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

    **When to Use**:
    - When you need to reduce the dimensionality of the data.
    - When dealing with high-dimensional data.
    """)
    st.code("""
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris

    # Load dataset
    iris = load_iris()
    X = iris.data

    # Initialize and fit model
    model = PCA(n_components=2)
    X_pca = model.fit_transform(X)

    # Plot
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, s=50, cmap='viridis')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
    """)

    # t-SNE
    st.subheader("t-SNE")
    st.write("""
    **Explanation**:
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that is particularly well-suited for visualizing high-dimensional data.

    **When to Use**:
    - When you need to visualize high-dimensional data in 2D or 3D.
    - When dealing with complex data structures.
    """)
    st.code("""
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits

    # Load dataset
    digits = load_digits()
    X = digits.data

    # Initialize and fit model
    model = TSNE(n_components=2, random_state=42)
    X_tsne = model.fit_transform(X)

    # Plot
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, s=50, cmap='viridis')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
    """)

    st.header("Association Rule Learning")

    # Apriori Algorithm
    st.subheader("Apriori Algorithm")
    st.write("""
    **Explanation**:
    The Apriori Algorithm is used for finding frequent itemsets in a dataset.

    **When to Use**:
    - Market basket analysis.
    - Recommendation systems.
    """)
    st.code("""
    from mlxtend.frequent_patterns import apriori, association_rules
    import pandas as pd

    # Sample dataset
    data = {'Transaction': [['Milk', 'Bread', 'Butter'],
                           ['Bread', 'Butter'],
                           ['Milk', 'Bread', 'Butter', 'Jam'],
                           ['Bread', 'Butter', 'Jam'],
                           ['Milk', 'Bread', 'Butter', 'Jam', 'Eggs']]}
    df = pd.DataFrame(data)

    # One-hot encoding
    one_hot = pd.get_dummies(df['Transaction'].apply(pd.Series).stack()).groupby(level=0).sum()

    # Initialize and fit model
    frequent_itemsets = apriori(one_hot, min_support=0.5, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Display rules
    print(rules)
    """)

    st.header("Anomaly Detection")

    # Isolation Forest
    st.subheader("Isolation Forest")
    st.write("""
    **Explanation**:
    Isolation Forest is an unsupervised anomaly detection algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

    **When to Use**:
    - Fraud detection.
    - Network intrusion detection.
    """)
    st.code("""
    from sklearn.ensemble import IsolationForest
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=0)

    # Initialize and fit model
    model = IsolationForest(contamination=0.1)
    model.fit(X)

    # Predict
    y_pred = model.predict(X)

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.show()
    """)

# Reinforcement Learning Page
def reinforcement_learning_page():
    st.title("Reinforcement Learning Algorithms")
    st.header("Model-Free Methods")

    # Q-Learning
    st.subheader("Q-Learning")
    st.write("""
    **Explanation**:
    Q-Learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state.

    **When to Use**:
    - When you need to learn optimal policies for decision-making tasks.
    - When dealing with environments where the dynamics are unknown.
    """)
    st.code("""
    import numpy as np
    import gym
    from gym import spaces

    # Define a simple environment
    class SimpleEnv(gym.Env):
        def __init__(self):
            super(SimpleEnv, self).__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Discrete(3)
            self.state = 0

        def step(self, action):
            if action == 0:
                self.state = (self.state + 1) % 3
            else:
                self.state = (self.state - 1) % 3
            reward = 1 if self.state == 2 else 0
            done = self.state == 2
            return self.state, reward, done, {}

        def reset(self):
            self.state = 0
            return self.state

    # Initialize environment
    env = SimpleEnv()

    # Initialize Q-table
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 1000

    # Q-Learning
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state

    # Display Q-table
    print(Q)
    """)

    # SARSA
    st.subheader("SARSA")
    st.write("""
    **Explanation**:
    SARSA (State-Action-Reward-State-Action) is a model-free reinforcement learning algorithm that learns the value of an action in a particular state.

    **When to Use**:
    - When you need to learn optimal policies for decision-making tasks.
    - When dealing with environments where the dynamics are unknown.
    """)
    st.code("""
    import numpy as np
    import gym
    from gym import spaces

    # Define a simple environment
    class SimpleEnv(gym.Env):
        def __init__(self):
            super(SimpleEnv, self).__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Discrete(3)
            self.state = 0

        def step(self, action):
            if action == 0:
                self.state = (self.state + 1) % 3
            else:
                self.state = (self.state - 1) % 3
            reward = 1 if self.state == 2 else 0
            done = self.state == 2
            return self.state, reward, done, {}

        def reset(self):
            self.state = 0
            return self.state

    # Initialize environment
    env = SimpleEnv()

    # Initialize Q-table
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 1000

    # SARSA
    for episode in range(episodes):
        state = env.reset()
        done = False
        action = np.argmax(Q[state]) if np.random.rand() > epsilon else env.action_space.sample()
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = np.argmax(Q[next_state]) if np.random.rand() > epsilon else env.action_space.sample()
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state = next_state
            action = next_action

    # Display Q-table
    print(Q)
    """)

    # Deep Q-Learning
    st.subheader("Deep Q-Learning (DQL)")
    st.write("""
    **Explanation**:
    Deep Q-Learning is a model-free reinforcement learning algorithm that uses a deep neural network to approximate the Q-values.

    **When to Use**:
    - When you need to learn optimal policies for decision-making tasks.
    - When dealing with environments where the dynamics are unknown.
    """)
    st.code("""
    import numpy as np
    import gym
    from gym import spaces
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    # Define a simple environment
    class SimpleEnv(gym.Env):
        def __init__(self):
            super(SimpleEnv, self).__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Discrete(3)
            self.state = 0

        def step(self, action):
            if action == 0:
                self.state = (self.state + 1) % 3
            else:
                self.state = (self.state - 1) % 3
            reward = 1 if self.state == 2 else 0
            done = self.state == 2
            return self.state, reward, done, {}

        def reset(self):
            self.state = 0
            return self.state

    # Initialize environment
    env = SimpleEnv()

    # Initialize Q-network
    model = Sequential([
        Dense(24, input_dim=env.observation_space.n, activation='relu'),
        Dense(24, activation='relu'),
        Dense(env.action_space.n, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # Hyperparameters
    gamma = 0.9
    epsilon = 0.1
    episodes = 1000

    # Deep Q-Learning
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(np.identity(env.observation_space.n)[state:state+1])
                action = np.argmax(q_values[0])
            next_state, reward, done, _ = env.step(action)
            target = reward + gamma * np.max(model.predict(np.identity(env.observation_space.n)[next_state:next_state+1])[0])
            target_f = model.predict(np.identity(env.observation_space.n)[state:state+1])
            target_f[0][action] = target
            model.fit(np.identity(env.observation_space.n)[state:state+1], target_f, epochs=1, verbose=0)
            state = next_state

    # Display Q-values
    print(model.predict(np.identity(env.observation_space.n)))
    """)

    st.header("Policy-Based Methods")

    # REINFORCE Algorithm
    st.subheader("REINFORCE Algorithm")
    st.write("""
    **Explanation**:
    REINFORCE is a policy-based reinforcement learning algorithm that directly optimizes the policy.

    **When to Use**:
    - When you need to learn optimal policies for decision-making tasks.
    - When dealing with environments where the dynamics are unknown.
    """)
    st.code("""
    import numpy as np
    import gym
    from gym import spaces
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    # Define a simple environment
    class SimpleEnv(gym.Env):
        def __init__(self):
            super(SimpleEnv, self).__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Discrete(3)
            self.state = 0

        def step(self, action):
            if action == 0:
                self.state = (self.state + 1) % 3
            else:
                self.state = (self.state - 1) % 3
            reward = 1 if self.state == 2 else 0
            done = self.state == 2
            return self.state, reward, done, {}

        def reset(self):
            self.state = 0
            return self.state

    # Initialize environment
    env = SimpleEnv()

    # Initialize policy network
    model = Sequential([
        Dense(24, input_dim=env.observation_space.n, activation='relu'),
        Dense(24, activation='relu'),
        Dense(env.action_space.n, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

    # Hyperparameters
    gamma = 0.9
    episodes = 1000

    # REINFORCE
    for episode in range(episodes):
        state = env.reset()
        done = False
        rewards = []
        states = []
        actions = []
        while not done:
            probs = model.predict(np.identity(env.observation_space.n)[state:state+1])
            action = np.random.choice(env.action_space.n, p=probs[0])
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
        discounts = np.logspace(0, len(rewards), num=len(rewards), base=gamma, endpoint=False)
        rewards = np.array(rewards) * discounts
        rewards = rewards[::-1].cumsum()[::-1]
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
        model.fit(np.identity(env.observation_space.n)[states], tf.keras.utils.to_categorical(actions, env.action_space.n), sample_weight=rewards, epochs=1, verbose=0)

    # Display policy
    print(model.predict(np.identity(env.observation_space.n)))
    """)

    st.header("Model-Based Methods")

    # Monte Carlo Tree Search (MCTS)
    st.subheader("Monte Carlo Tree Search (MCTS)")
    st.write("""
    **Explanation**:
    MCTS is a heuristic search algorithm for decision processes, most notably used in decision processes like games.

    **When to Use**:
    - When you need to learn optimal policies for decision-making tasks.
    - When dealing with environments where the dynamics are known.
    """)
    st.code("""
    import numpy as np
    import gym
    from gym import spaces

    # Define a simple environment
    class SimpleEnv(gym.Env):
        def __init__(self):
            super(SimpleEnv, self).__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Discrete(3)
            self.state = 0

        def step(self, action):
            if action == 0:
                self.state = (self.state + 1) % 3
            else:
                self.state = (self.state - 1) % 3
            reward = 1 if self.state == 2 else 0
            done = self.state == 2
            return self.state, reward, done, {}

        def reset(self):
            self.state = 0
            return self.state

    # Initialize environment
    env = SimpleEnv()

    # MCTS
    class Node:
        def __init__(self, state, parent=None, action=None):
            self.state = state
            self.parent = parent
            self.action = action
            self.children = []
            self.visits = 0
            self.value = 0

        def is_fully_expanded(self):
            return len(self.children) == env.action_space.n

        def best_child(self, c_param=1.0):
            choices_weights = [
                (child.value / (child.visits + 1)) + c_param * np.sqrt((2 * np.log(self.visits + 1)) / (child.visits + 1))
                for child in self.children
            ]
            return self.children[np.argmax(choices_weights)]

        def most_visited_child(self):
            return self.children[np.argmax([child.visits for child in self.children])]

    def select(node):
        while not node.is_fully_expanded():
            node = node.best_child()
        return node

    def expand(node):
        tried_actions = [child.action for child in node.children]
        new_state = env.step(np.random.choice(list(set(range(env.action_space.n)) - set(tried_actions)))[0])[0]
        node.children.append(Node(new_state, parent=node, action=new_state))
        return node.children[-1]

    def simulate(node):
        current_reward = 0
        current_state = node.state
        while not env.step(np.random.choice(env.action_space.n))[2]:
            current_reward += env.step(np.random.choice(env.action_space.n))[1]
        return current_reward

    def backpropagate(node, result):
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def mcts(root):
        for _ in range(1000):
            node = select(root)
            if not node.is_fully_expanded():
                node = expand(node)
            result = simulate(node)
            backpropagate(node, result)
        return root.most_visited_child().action

    # Initialize root node
    root = Node(env.reset())

    # MCTS
    action = mcts(root)
    print(action)
    """)

# Semi-Supervised Learning Page
def semi_supervised_learning_page():
    st.title("Semi-Supervised Learning Algorithms")
    st.header("Self-Training")

    # Self-Training
    st.subheader("Self-Training")
    st.write("""
    **Explanation**:
    Self-Training is a semi-supervised learning algorithm that uses the model to label unlabeled data.

    **When to Use**:
    - When you have a mix of labeled and unlabeled data.
    - When you need to improve the performance of a model with limited labeled data.
    """)
    st.code("""
    from sklearn.semi_supervised import SelfTrainingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

    # Split data into labeled and unlabeled
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.8, random_state=42)

    # Initialize and fit model
    model = SelfTrainingClassifier(LogisticRegression())
    model.fit(X_labeled, y_labeled)

    # Predict
    y_pred = model.predict(X_unlabeled)

    # Evaluate
    accuracy = accuracy_score(y_unlabeled, y_pred)
    print(f'Accuracy: {accuracy}')
    """)

    st.header("Co-Training")

    # Co-Training
    st.subheader("Co-Training")
    st.write("""
    **Explanation**:
    Co-Training is a semi-supervised learning algorithm that trains multiple models with different features.

    **When to Use**:
    - When you have a mix of labeled and unlabeled data.
    - When you need to improve the performance of a model with limited labeled data.
    """)
    st.code("""
    from sklearn.semi_supervised import SelfTrainingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

    # Split data into labeled and unlabeled
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.8, random_state=42)

    # Initialize and fit model
    model = SelfTrainingClassifier(LogisticRegression())
    model.fit(X_labeled, y_labeled)

    # Predict
    y_pred = model.predict(X_unlabeled)

    # Evaluate
    accuracy = accuracy_score(y_unlabeled, y_pred)
    print(f'Accuracy: {accuracy}')
    """)

# Extended Categories Page
def extended_categories_page():
    st.title("Extended Categories")
    st.header("Deep Learning Algorithms")

    # Convolutional Neural Networks (CNNs)
    st.subheader("Convolutional Neural Networks (CNNs)")
    st.write("""
    **Explanation**:
    CNNs are a type of deep learning algorithm used primarily for image data.

    **When to Use**:
    - Image classification.
    - Object detection.
    """)
    st.code("""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Initialize and fit model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    """)

    # Recurrent Neural Networks (RNNs)
    st.subheader("Recurrent Neural Networks (RNNs)")
    st.write("""
    **Explanation**:
    RNNs are a type of deep learning algorithm used primarily for sequence data.

    **When to Use**:
    - Time series forecasting.
    - Natural language processing.
    """)
    st.code("""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence

    # Load dataset
    max_features = 10000
    maxlen = 500
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    # Initialize and fit model
    model = Sequential([
        SimpleRNN(128, input_shape=(maxlen, 1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    """)

    # Long Short-Term Memory (LSTM) Networks
    st.subheader("Long Short-Term Memory (LSTM) Networks")
    st.write("""
    **Explanation**:
    LSTM networks are a type of RNN used for sequence data with long-term dependencies.

    **When to Use**:
    - Time series forecasting.
    - Natural language processing.
    """)
    st.code("""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence

    # Load dataset
    max_features = 10000
    maxlen = 500
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    # Initialize and fit model
    model = Sequential([
        LSTM(128, input_shape=(maxlen, 1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    """)

    # Generative Adversarial Networks (GANs)
    st.subheader("Generative Adversarial Networks (GANs)")
    st.write("""
    **Explanation**:
    GANs are a type of deep learning algorithm used for generative modeling.

    **When to Use**:
    - Image generation.
    - Data augmentation.
    """)
    st.code("""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    import numpy as np

    # Generate synthetic data
    def generate_real_samples(n):
        X = np.random.rand(n, 28, 28, 1)
        y = np.ones((n, 1))
        return X, y

    def generate_fake_samples(generator, n):
        X = np.random.rand(n, 100)
        X = generator.predict(X)
        y = np.zeros((n, 1))
        return X, y

    # Define generator model
    def define_generator(latent_dim):
        model = Sequential()
        model.add(Dense(128 * 7 * 7, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
        return model

    # Define discriminator model
    def define_discriminator(in_shape=(28, 28, 1)):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    # Define GAN model
    def define_gan(generator, discriminator):
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return model

    # Initialize models
    latent_dim = 100
    generator = define_generator(latent_dim)
    discriminator = define_discriminator()
    gan = define_gan(generator, discriminator)

    # Train GAN
    def train_gan(generator, discriminator, gan, latent_dim, n_epochs=10000, n_batch=128):
        for epoch in range(n_epochs):
            X_real, y_real = generate_real_samples(n_batch)
            X_fake, y_fake = generate_fake_samples(generator, n_batch)
            discriminator.train_on_batch(X_real, y_real)
            discriminator.train_on_batch(X_fake, y_fake)
            X_gan = np.random.rand(n_batch, latent_dim)
            y_gan = np.ones((n_batch, 1))
            gan.train_on_batch(X_gan, y_gan)
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}')

    train_gan(generator, discriminator, gan, latent_dim)
    """)

    st.header("Ensemble Learning Algorithms")

    # Bagging
    st.subheader("Bagging")
    st.write("""
    **Explanation**:
    Bagging is an ensemble learning method that combines multiple models to improve the accuracy and control over-fitting.

    **When to Use**:
    - When you need a robust model with high accuracy.
    - When dealing with large datasets.
    """)
    st.code("""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    """)

    # Boosting
    st.subheader("Boosting")
    st.write("""
    **Explanation**:
    Boosting is an ensemble learning method that combines multiple weak models to create a strong model.

    **When to Use**:
    - When you need a robust model with high accuracy.
    - When dealing with large datasets.
    """)
    st.code("""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    """)

    # Stacking
    st.subheader("Stacking")
    st.write("""
    **Explanation**:
    Stacking is an ensemble learning method that combines predictions from multiple models.

    **When to Use**:
    - When you need a robust model with high accuracy.
    - When dealing with large datasets.
    """)
    st.code("""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit model
    model = StackingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('svc', SVC())
    ], final_estimator=LogisticRegression())
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    """)

# Main function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Index", "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Semi-Supervised Learning", "Extended Categories"])

    if page == "Index":
        index_page()
    elif page == "Supervised Learning":
        supervised_learning_page()
    elif page == "Unsupervised Learning":
        unsupervised_learning_page()
    elif page == "Reinforcement Learning":
        reinforcement_learning_page()
    elif page == "Semi-Supervised Learning":
        semi_supervised_learning_page()
    elif page == "Extended Categories":
        extended_categories_page()

if __name__ == "__main__":
    main()
