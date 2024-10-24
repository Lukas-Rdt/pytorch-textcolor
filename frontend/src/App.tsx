import { useEffect, useState } from "react";
import "./App.css";
import ColorPalette from "./ColorPalette";

interface ColorResponse {
  prediction: number;
}

function App() {
  const baseURL: string = "http://127.0.0.1:5000";
  const [color, setColor] = useState({ r: 0, g: 0, b: 0 });
  const [prediction, setPrediction] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isFeedbackLoading, setIsFeedbackLoading] = useState(false);
  const [showPrediction, setShowPrediction] = useState(true);

  const generateRandomColor = () => {
    const newColor = {
      r: Math.random(),
      g: Math.random(),
      b: Math.random(),
    };
    setColor(newColor);
    return newColor;
  };

  const generateColor = () => {
    const newColor = generateRandomColor();
    getColorPrediction(newColor);
  };

  const togglePrediction = () => {
    if (showPrediction) {
      setShowPrediction(false);
    } else {
      setShowPrediction(true);
    }
  };

  const getColorPrediction = async (rgbColor: {
    r: number;
    g: number;
    b: number;
  }) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${baseURL}/api/get-color`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          rgb: [rgbColor.r, rgbColor.g, rgbColor.b],
        }),
      });
      const data: ColorResponse = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      console.error("Error getting color prediction:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const updateNeuralNetwork = async () => {
    try {
      setIsTraining(true);
      await fetch(`${baseURL}/api/update-neural-network`, {
        method: "POST",
      });
    } catch (error) {
      console.error("Error updating neural network:", error);
    } finally {
      setIsTraining(false);
    }
  };

  const submitFeedback = async (shouldBeWhite: boolean) => {
    try {
      setIsFeedbackLoading(true);
      const response = await fetch(`${baseURL}/api/add-training-data`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          rgb: [color.r, color.g, color.b],
          output: shouldBeWhite ? 1 : 0,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit Feedback");
      }

      await updateNeuralNetwork();
    } catch (error) {
      console.log("Error submitting Feedback:", error);
    } finally {
      generateColor();
      setIsFeedbackLoading(false);
    }
  };

  useEffect(() => {
    const initialColor = generateRandomColor();
    getColorPrediction(initialColor);
  }, []);

  const rgbString = `rgb(${Math.round(color.r * 255)}, ${Math.round(
    color.g * 255
  )}, ${Math.round(color.b * 255)})`;

  return (
    <div className="flex flex-col items-center gap-6 p-8">
      <div
        className="w-64 h-64 flex flex-col items-center justify-center p-4 relative"
        style={{ backgroundColor: rgbString }}>
        <p className="text-white font-bold mb-2">Weißer Text</p>

        <p className="text-black font-bold mb-2">Schwarzer Text</p>

        {showPrediction && (
          <p
            className={`font-bold ${
              prediction !== null && prediction > 0.5
                ? "text-white"
                : "text-black"
            }`}>
            {isLoading ? "Lade..." : "Dynamischer Text"}
          </p>
        )}
      </div>

      <div className="flex flex-col items-center gap-4">
        <div className="flex gap-4">
          <button
            onClick={generateColor}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors">
            Neue Farbe
          </button>

          <button onClick={togglePrediction}>Toggle prediction</button>

          <button
            onClick={updateNeuralNetwork}
            disabled={isTraining}
            className={`px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors ${
              isTraining ? "opacity-50 cursor-not-allowed" : ""
            }`}>
            {isTraining ? "Training läuft..." : "Neural Network trainieren"}
          </button>
        </div>

        <div className="flex gap-4 mt-4">
          <button
            onClick={() => submitFeedback(false)}
            disabled={isFeedbackLoading}
            className="px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-900 transition-colors disabled:opacity-50">
            Sollte schwarzer Text sein
          </button>

          <button
            onClick={() => submitFeedback(true)}
            disabled={isFeedbackLoading}
            className="px-4 py-2 bg-gray-200 text-black rounded hover:bg-gray-300 transition-colors disabled:opacity-50">
            Sollte weißer Text sein
          </button>
        </div>
      </div>

      {prediction !== null && (
        <p className="text-sm text-gray-600">
          Vorhersage: {showPrediction && prediction.toFixed(5)}
        </p>
      )}

      {isFeedbackLoading && (
        <p className="text-sm text-gray-600">Feedback wird verarbeitet...</p>
      )}
      <ColorPalette />
    </div>
  );
}

export default App;
