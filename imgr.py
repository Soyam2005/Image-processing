import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ---------------------------------------------------------
# 1. Generate and Prepare the Sample Dataset
# ---------------------------------------------------------
# Create 1000 data points of two interlocking moons with slight noise
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)


# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (Neural Networks perform much better when inputs are scaled around 0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. Build the Artificial Neural Network
# ---------------------------------------------------------
model = Sequential([
    Input(shape=(2,)),
    # Input Layer & Hidden Layer 1: 16 neurons. 
    # 'relu' introduces the non-linearity needed to bend the decision boundary.
    Dense(16, activation='relu'),
    
    # Hidden Layer 2: 8 neurons for further pattern refinement.
    Dense(8, activation='relu'),
    
    # Output Layer: 1 neuron. 
    # 'sigmoid' squashes the output to a probability between 0 and 1.
    Dense(1, activation='sigmoid')
])

# ---------------------------------------------------------
# 3. Compile and Train the Model
# ---------------------------------------------------------
# 'adam' is the optimizer (handles the backpropagation math)
# 'binary_crossentropy' is the standard loss function for two-class problems
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training the ANN...")
# Train the model over 50 epochs (iterations over the entire dataset)
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate on the unseen test data
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

# ---------------------------------------------------------
# 4. Visualize the Detected Pattern (Decision Boundary)
# ---------------------------------------------------------
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict over the entire grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
    plt.colorbar(label='Network Confidence (Probability)')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors='k', s=40)
    plt.title("ANN Pattern Detection: Learned Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

print("Generating visualization...")
plot_decision_boundary(X_test_scaled, y_test, model)

# ---------------------------------------------------------
# 5. Predict Class Labels for New Samples
# ---------------------------------------------------------
print("\nPredicting class labels for new custom samples...")
# Create a few new data points
new_samples = np.array([[0.5, 0.5], [-0.5, -0.5], [1.0, -1.0]])

# We must scale them using the same scaler used for training
new_samples_scaled = scaler.transform(new_samples)

# Predict probabilities
predictions_prob = model.predict(new_samples_scaled, verbose=0)

# Convert probabilities to class labels (0 or 1) using 0.5 as threshold
predictions_labels = (predictions_prob > 0.5).astype(int)

for i in range(len(new_samples)):
    print(f"Sample: {new_samples[i]} -> Probability: {predictions_prob[i][0]:.4f} => Predicted Class: {predictions_labels[i][0]}")
