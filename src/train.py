from config import *
from preprocessing import *
from model import *
import joblib

def main():
    # Load data
    df = load_data(TRAIN_PATH)

    # Add RUL
    df = add_rul(df)

    # Scale
    df, scaler = scale_data(df)

    # Save scaler
    joblib.dump(scaler, "results/scaler.pkl")

    # Create sequences
    X, y = create_sequences(df, SEQUENCE_LENGTH)

    print("Training Shape:", X.shape)

    # Build model
    model = build_model((SEQUENCE_LENGTH, X.shape[2]))

    # Train
    model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )

    # Save model
    model.save(MODEL_SAVE_PATH)

    print("Training Completed & Model Saved")

if __name__ == "__main__":
    main()