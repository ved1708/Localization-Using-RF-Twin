
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def load_dataset(filename="rf_dataset.pkl"):
    """Loads the RF dataset from a pickle file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset with {len(dataset)} samples.")
    return dataset

def create_feature_vector(features):
    """Creates a feature vector from the extracted CIR features."""
    path_gains = features['path_gains']
    path_delays = features['path_delays']

    # Use log-power from gains
    power = 10 * np.log10(path_gains**2 + 1e-12)
    delays = path_delays.real

    max_paths = 20
    if len(power) > max_paths:
        power = power[:max_paths]
        delays = delays[:max_paths]
    else:
        power = np.pad(power, (0, max_paths - len(power)), 'constant', constant_values=-100)
        delays = np.pad(delays, (0, max_paths - len(delays)), 'constant', constant_values=0)
        
    return np.concatenate([power, delays])

def prepare_data(dataset):
    """Prepares training and testing data."""
    positions = np.array([d['position'] for d in dataset])
    feature_vectors = np.array([create_feature_vector(d['features']) for d in dataset])

    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, positions, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")
    print(f"Feature vector shape: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_nn_model(input_shape):
    """MLP model for localization."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)  # Output layer for 3D coordinates (x, y, z)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error')
    return model

def evaluate_performance(y_true, y_pred):
    """Performance metrics."""
    errors = np.linalg.norm(y_true - y_pred, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    print("\n--- Performance Metrics ---")
    print(f"RMSE: {rmse:.4f} m")
    print(f"Mean Error: {mean_error:.4f} m")
    print(f"Median Error: {median_error:.4f} m")
    print(f"Std Dev of Error: {std_error:.4f} m")
    print("---------------------------\n")
    return errors, rmse

def plot_results(errors, y_true, y_pred, filename="nn_localization_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Neural Network Localization Performance', fontsize=16)

    # 1. CDF of Localization Error
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[0].plot(sorted_errors, cdf, linewidth=2)
    axes[0].set_title('CDF of Localization Error')
    axes[0].set_xlabel('Error (m)')
    axes[0].set_ylabel('Probability')
    axes[0].grid(True, linestyle='--')
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)

    # 2. Histogram of Errors
    axes[1].hist(errors, bins=30, density=True, alpha=0.7, label='Error Distribution')
    mean_error = np.mean(errors)
    axes[1].axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error = {mean_error:.2f}m')
    axes[1].set_title('Error Distribution')
    axes[1].set_xlabel('Error (m)')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, linestyle='--')
    
    # 3. Spatial Error Map
    sc = axes[2].scatter(y_true[:, 0], y_true[:, 1], c=errors, cmap='viridis', alpha=0.8, vmin=0, vmax=np.percentile(errors, 95))
    plt.colorbar(sc, ax=axes[2], label='Localization Error (m)')
    axes[2].set_title('Spatial Distribution of Errors (Top-Down View)')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].set_aspect('equal', 'box')
    axes[2].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print(f"Results plot saved to {filename}")

def main():
    try:
        # Load and prepare data
        dataset = load_dataset("../rf_dataset.pkl")
        X_train, X_test, y_train, y_test, scaler = prepare_data(dataset)

        # Build and train the model
        model = build_nn_model(X_train.shape[1])
        print("Training the neural network model...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
        )
        print("Model training complete.")

        y_pred = model.predict(X_test)
        errors, rmse = evaluate_performance(y_test, y_pred)

        np.savetxt("mlp_errors.csv", errors, delimiter=",")
        plot_results(errors, y_test, y_pred)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()