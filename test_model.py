import keras
from pathlib import Path

MODEL_PATH = Path(
    r"C:\Users\Arham Bhatti\Documents\re_id_enhance\CycleGAN\outputs\G_L2N.keras"
)

print("MODEL PATH:")
print(MODEL_PATH)

print("FILE EXISTS:")
print(MODEL_PATH.exists())

try:
    model = keras.models.load_model(str(MODEL_PATH))
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("ERROR:")
    print(e)
