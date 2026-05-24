import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Functions from existing scripts
from evaluate_localization import load_dataset as load_knn_dataset, prepare_data as prepare_knn_data, knn_localization, evaluate_performance as eval_knn
from train_nn_localizer import load_dataset as load_nn_dataset, prepare_data as prepare_nn_data, build_nn_model, evaluate_performance as eval_nn

def main():
    # 1. kNN (k=10)
    print("--- Running kNN (k=10) ---")
    knn_data = load_knn_dataset('../rf_dataset.pkl')
    np.random.seed(42)
    knn_train_pos, knn_train_feat, knn_test_pos, knn_test_feat = prepare_knn_data(knn_data, test_split=0.2)
    knn_pred = knn_localization(knn_train_pos, knn_train_feat, knn_test_feat, k=10)
    knn_metrics = eval_knn(knn_test_pos, knn_pred)
    knn_errors = knn_metrics['errors']

    # 2. MLP
    print("\n--- Running MLP ---")
    nn_data = load_nn_dataset('../rf_dataset.pkl')
    X_train, X_test, y_train, y_test, scaler = prepare_nn_data(nn_data)
    model = build_nn_model(X_train.shape[1])
    
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
    )
    y_pred = model.predict(X_test, verbose=0)
    nn_errors, _ = eval_nn(y_test, y_pred)

    # 3. Plot CDFs
    print("\n--- Plotting Comparison ---")
    plt.figure(figsize=(10, 6))
    
    # kNN CDF
    sorted_knn = np.sort(knn_errors)
    cdf_knn = np.arange(1, len(sorted_knn) + 1) / len(sorted_knn)
    plt.plot(sorted_knn, cdf_knn, linewidth=2, label='kNN (k=10)')
    
    # MLP CDF
    sorted_nn = np.sort(nn_errors)
    cdf_nn = np.arange(1, len(sorted_nn) + 1) / len(sorted_nn)
    plt.plot(sorted_nn, cdf_nn, linewidth=2, label='MLP / NN', linestyle='--')
    
    plt.title('CDF Comparison: kNN vs MLP')
    plt.xlabel('Localization Error (m)')
    plt.ylabel('CDF')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    
    output_path = 'cdf_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
