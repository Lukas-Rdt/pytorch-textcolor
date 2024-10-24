from flask import Flask, jsonify, request
from flask_cors import CORS
from text_color_net import NeuralNetwork
from color_palette_net import ColorPaletteNetwork

app = Flask(__name__)
CORS(app)

nn = NeuralNetwork()
palette_nn = ColorPaletteNetwork()

@app.route('/api/update-neural-network', methods=['POST'])
def update_nn():
    try:
        result = nn.train()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-color', methods=['POST'])
def get_color():
    try:
        data = request.get_json()
        if not data or "rgb" not in data:
            return jsonify({"error": "No RGB values provided"}), 400
        rgb_values = data["rgb"]
        result = nn.predict(rgb_values)
        return jsonify({"prediction": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-training-data', methods=["POST"])
def add_training_data():
    try:
        data = request.get_json()
        if not data or "rgb" not in data or "output" not in data:
            return jsonify({"error": "Missing rgb or output values"}), 400
        
        rgb_values = data["rgb"]
        output = data["output"]

        if not all(0 <= x <= 1 for x in rgb_values) or len(rgb_values) != 3:
            return jsonify({"error": "RGB values between 0 and 1"}), 400
        if output not in [0, 1]:
            return jsonify({"error": "Output must be 0 or 1"}), 400
        
        success = nn.add_training_data(rgb_values, output)

        if success:
            return jsonify({"message": "Training data added successfully"})
        else:
            return jsonify({"error": "Training data failed to add"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# color palette generation apis

@app.route("/api/train-palette-network", methods=["POST"])
def train_palette_network():
    palette_nn.train()
    try:
        result = palette_nn.train()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/update-palette-network", methods=["POST"])
def update_palette_network():
    try:
        data = request.get_json()
        print(data["input_rgb"])
        network = ColorPaletteNetwork()
        result = network.add_training_data(data["input_rgb"], data["output_palette"])
        return jsonify({"palette": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate-palette", methods=["POST"])
def generate_palette():
    try:
        data = request.get_json()
        if not data or "rgb" not in data:
            return jsonify({"error": "No RGB values provided"})
        rgb_values = data["rgb"]
        result = palette_nn.generate_palette(rgb_values)
        return jsonify({"palette": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/generate-palette-data", methods=["POST"])
def generate_palette_data():
    try:
        data = request.get_json()
        if not data or "rgb" not in data:
            return jsonify({"error": "No RGB values provided"})
        rgb_values = data["rgb"]
        result = palette_nn.generate_training_variations(rgb_values)
        return jsonify({"palette": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)