from flask import Flask, request, jsonify
import torch
import time # Keep time for timestamping if needed

from models.drone_agent_model import DroneModel
from data_models.drone_data_model import DroneData
from config import hyperparameters as hp

app = Flask(__name__)

# --- Global Model and Device --- (Load once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DroneModel(
    fusion_dim=hp.FUSION_DIM,
    scalar_input_dim=hp.SCALAR_INPUT_DIM,
    vit_image_size=hp.IMG_SIZE,
    vit_patch_size=hp.VIT_PATCH_SIZE,
    num_control_outputs=hp.NUM_CONTROL_OUTPUTS,
    learning_rate=hp.LEARNING_RATE # Learning rate isn't used for inference but model expects it
).to(device)
model.eval() # Set to evaluation mode

# --- (Optional) Load Pre-trained Weights --- 
# try:
#     model.load_state_dict(torch.load("path_to_your_trained_model.pth", map_location=device))
#     print("Pre-trained model weights loaded.")
# except FileNotFoundError:
#     print("No pre-trained model weights found, using initialized model.")
# except Exception as e:
#     print(f"Error loading model weights: {e}")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Drone control API is running."}), 200

@app.route('/predict', methods=['POST'])
def predict_controls():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    try:
        drone_data_instance = DroneData.from_dict(data)
    except KeyError as e:
        return jsonify({"error": f"Missing data field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid data format: {e}"}), 400

    with torch.no_grad():
        predicted_outputs = model(drone_data_instance)
    
    response = {
        "predicted_controls": predicted_outputs.squeeze().tolist(),
        "timestamp": time.time()
    }
    return jsonify(response), 200

if __name__ == '__main__':
    print(f"Starting Flask server on device: {device}")
    app.run(debug=True, host='0.0.0.0', port=5600)