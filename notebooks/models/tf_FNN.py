import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from data_processing import process_dataframe


class BestModelTrainer:
    def __init__(self, df_train_scaled_range, y_train_transformed, df_val_scaled_range, y_val_transformed):
        self.df_train_scaled_range = df_train_scaled_range
        self.y_train_transformed = y_train_transformed
        self.df_val_scaled_range = df_val_scaled_range
        self.y_val_transformed = y_val_transformed

        # best hyperparameters for model, we got this from tuning
        # See tf_FNN_tuning.ipynb
        self.best_hyperparameters = {
            'num_layers': 3,
            'units_0': 32, 
            'activation': 'relu',
            'dropout': 0.2,  # combat overfitting
            'units_1': 64,
            'units_2': 32 }

        # input shape, taken from the training data shape
        self.input_shape = self.df_train_scaled_range.shape[1]

    def build_model(self):
        """
        Builds the model using the parameters above. It's a simple 3-layer network 
        with dropout for regularization.
        """
        
        model = keras.Sequential()

        # The first dense layer, 32 units, relu activation
        model.add(layers.Dense(self.best_hyperparameters['units_0'], activation=self.best_hyperparameters['activation'], input_shape=(self.input_shape,)))
        model.add(layers.Dropout(self.best_hyperparameters['dropout']))  # Adding dropout to prevent overfitting

        # Second dense layer, 64 units, still using relu
        model.add(layers.Dense(self.best_hyperparameters['units_1'], activation=self.best_hyperparameters['activation']))
        model.add(layers.Dropout(self.best_hyperparameters['dropout']))  # More dropout

        # Third dense layer, back down to 32 units
        model.add(layers.Dense(self.best_hyperparameters['units_2'], activation=self.best_hyperparameters['activation']))
        model.add(layers.Dropout(self.best_hyperparameters['dropout']))  # Another dropout layer

        # Output layer, egression
        model.add(layers.Dense(1))  # Linear output, no activation function for regression

        # Compiling the model with Adam optimizer and MSE loss function
        model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mae'])
        return model


    def train_model(self):
        """
        Trains the model using the training data and evaluates it on the validation set. 
        The training will stop early if the validation loss does not improve.
        """
        model = self.build_model()

        history = model.fit(
            self.df_train_scaled_range, self.y_train_transformed,  # Input training data
            validation_data=(self.df_val_scaled_range, self.y_val_transformed),
            epochs=50,
            batch_size=32,
            # early stopping
            callbacks=[EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)],
            verbose=1  # Show progress during training
            #verbose=False 
        )

        return model, history


def main():
    """
    The main function that loads the data, sets up the trainer, trains the model, 
    and evaluates it. Model and training history will also be saved.
    """
    (df_train_scaled_range, df_val_scaled_range, df_test_scaled_range,
        y_train_transformed, y_val_transformed, y_test_transformed
        ) = process_dataframe()
  
    # Initialise the model trainer class
    model_trainer = BestModelTrainer(df_train_scaled_range, y_train_transformed, df_val_scaled_range, y_val_transformed)

    best_model, history = model_trainer.train_model()

    # Evaluate
    test_loss, test_mae = best_model.evaluate(df_val_scaled_range, y_val_transformed)
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Saving model
    best_model.save('model_tf+results/best_model.h5')

    # Saving training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history.csv', index=False)  # Save the training progress to CSV
    print("\nTraining history saved as 'training_history.csv'.")

if __name__ == "__main__":
    main()
