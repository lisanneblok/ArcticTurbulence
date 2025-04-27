import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, X_val, y_val, qt, 
                            activ='tanh', opt='adam', n_epochs=1000, 
                            pat=30, val_split=0.15, n_units1=35, 
                            n_units2=35, batch_size=16, SEED=42):
    """
    Train an MLPRegressor, make predictions, and evaluate performance, including handling class imbalance.
    
    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target set.
        X_val (pd.DataFrame): Validation feature set.
        y_val (pd.Series): Validation target set.
        qt (QuantileTransformer): Quantile transformer for inverse transformation.
        activ (str): Activation function for MLP.
        opt (str): Optimizer for MLP.
        n_epochs (int): Number of epochs for MLP.
        pat (int): Patience for early stopping.
        val_split (float): Validation fraction for early stopping.
        n_units1 (int): Number of neurons in first hidden layer.
        n_units2 (int): Number of neurons in second hidden layer.
        batch_size (int): Batch size for training.
        SEED (int): Random seed for reproducibility.
    
    Returns:
        dict: Model evaluation metrics (R², MSE, MAE) and learning rate curve.
    """

    # MLP hyperparameters
    activ = 'tanh'
    opt = 'adam'
    pat = 30
    n_epochs = 1000
    val_split = 0.15
    dropout_fraction = 0.2 
    n_units1 = 35
    n_units2 = 35
    batch_size = 16
    
    # Train MLPRegressor
    regr = MLPRegressor(hidden_layer_sizes=(n_units1, n_units2),
                        activation=activ,
                        solver=opt,
                        max_iter=n_epochs,
                        random_state=SEED,
                        early_stopping=True,
                        n_iter_no_change=pat,
                        validation_fraction=val_split,
                        batch_size=batch_size,
                        warm_start=True,
                        verbose=False)

    # Fit model and keep track of learning curves (loss history)
    loss_curve = []
    for epoch in range(n_epochs):
        regr.fit(X_train, y_train)
        loss_curve.append(regr.loss_)
        
        # Early stopping if the loss curve does not improve
        if len(loss_curve) > pat and loss_curve[-1] >= min(loss_curve[-pat-1:]):
            break
    
    # Predict
    y_pred_transformed = regr.predict(X_test)
    y_pred_transformed = y_pred_transformed.reshape(-1, 1)

    # Inverse transform predictions to original log_eps space
    y_pred = pd.DataFrame(qt.inverse_transform(y_pred_transformed), columns=['log_eps'])

    # Evaluate
    u = ((y_test.flatten() - y_pred['log_eps'].flatten()) ** 2).sum()
    v = ((y_test.flatten() - y_test.flatten().mean()) ** 2).sum()
    r2_score = 1 - (u/v)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Plot learning rate curve (loss curve)
    plt.figure(figsize=(8, 6))
    plt.plot(loss_curve, label='Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Rate and Loss Curve')
    plt.legend()
    plt.show()
    
    # Return evaluation metrics and the learning curve
    return {
        "R²": r2_score,
        "MSE": mse,
        "MAE": mae,
        "Loss_Curve": loss_curve
    }
