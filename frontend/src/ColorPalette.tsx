import { useState } from "react";

const ColorPalette = () => {
  const baseURL: string = "http://127.0.0.1:5000";
  const [baseColor, setBaseColor] = useState<string>("#808080");
  const [palette, setPalette] = useState<string[]>(Array(10).fill("#808080"));
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [selectedColorIndex, setSelectedColorIndex] = useState<number | null>(
    null
  );

  const hexToRgb = (hex: string): number[] => {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    return [r, g, b];
  };

  const rgbToHex = (rgb: number[]): string => {
    const toHex = (n: number) => {
      const hex = Math.round(n * 255).toString(16);
      return hex.length === 1 ? "0" + hex : hex;
    };
    return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`;
  };

  const validateAndNormalizePalette = (colors: string[]): string[] => {
    const normalized = [...colors];

    // Fülle mit Standardfarben auf
    while (normalized.length < 10) {
      normalized.push("#808080");
    }

    // Schneide überzählige Farben ab
    return normalized.slice(0, 10);
  };

  const handleColorClick = (index: number) => {
    setSelectedColorIndex(index);
  };

  const handleColorChange = (Color: string) => {
    if (selectedColorIndex !== null) {
      const newPalette = [...palette];
      newPalette[selectedColorIndex] = Color;
      setPalette(newPalette);
    }
  };

  const generateTrainingVariations = async () => {
    try {
      setError("");
      setLoading(true);
      const rgb = hexToRgb(baseColor);
      const response = await fetch(`${baseURL}/api/generate-palette-data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rgb }),
      });

      if (!response.ok)
        throw new Error("Failed to generate palette training variations");

      const data = await response.json();
      // Stelle sicher, dass wir immer 10 Farben haben
      const normalizedPalette = validateAndNormalizePalette(
        data.palette.map(rgbToHex)
      );
      setPalette(normalizedPalette);
    } catch (error) {
      setError("Failed to generate variations.");
      console.error("Error generating variations:", error);
    } finally {
      setLoading(false);
    }
  };

  const updateNetwork = async () => {
    try {
      setError("");
      setLoading(true);
      const rgb = hexToRgb(baseColor);
      const variations = palette.map(hexToRgb);

      if (variations.length !== 10) {
        throw new Error("Palette must contain exactly 10 colors");
      }

      console.log("Sending data:", {
        input_rgb: rgb,
        output_palette: variations,
      });

      const response = await fetch(`${baseURL}/api/update-palette-network`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_rgb: rgb,
          output_palette: variations,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to update Network");
      }
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Failed to update network"
      );
      console.error("Error updating network:", error);
    } finally {
      setLoading(false);
    }
  };

  const trainNetwork = async () => {
    try {
      setError("");
      setLoading(true);

      await fetch(`${baseURL}/api/train-palette-network`, {
        method: "POST",
      });
    } catch (error) {
      setError("Failed to train network.");
      console.error("Error train network:", error);
    } finally {
      setLoading(false);
    }
  };

  const generatePalette = async () => {
    try {
      setError("");
      setLoading(true);
      const rgb = hexToRgb(baseColor);

      const response = await fetch(`${baseURL}/api/generate-palette`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rgb }),
      });

      if (!response.ok) throw new Error("Failed to generate palette");

      const data = await response.json();
      const normalizedPalette = validateAndNormalizePalette(
        data.palette.map(rgbToHex)
      );
      setPalette(normalizedPalette);
    } catch (error) {
      setError("Failed to generate palette.");
      console.error("Error generating palette:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex gap-8 p-6">
      {/* Linke Spalte mit Farbblöcken */}
      <div className="flex flex-col gap-2">
        {palette.map((color, index) => (
          <div
            key={index}
            onClick={() => handleColorClick(index)}
            className={`w-32 h-10 rounded-md border cursor-pointer transition-all duration-200 ${
              selectedColorIndex === index
                ? "ring-2 ring-blue-500 scale-105"
                : "hover:scale-105"
            }`}
            style={{ backgroundColor: color }}
          />
        ))}
      </div>

      {/* Rechte Spalte mit Kontrollen */}
      <div className="p-6 space-y-6">
        <div className="space-y-2">
          <label className="block text-sm font-medium">Base Color</label>
          <input
            type="color"
            value={baseColor}
            onChange={(e) => setBaseColor(e.target.value)}
            className="w-full h-10"
          />
        </div>

        {selectedColorIndex !== null && (
          <div className="space-y-2">
            <label className="block text-sm font-medium">
              Selected Color (Position {selectedColorIndex + 1})
            </label>
            <input
              type="color"
              value={palette[selectedColorIndex]}
              onChange={(e) => handleColorChange(e.target.value)}
              className="w-full h-10"
            />
          </div>
        )}

        {error && (
          <div className="my-4">
            <strong>{error}</strong>
          </div>
        )}

        <div className="space-y-4">
          <button
            onClick={generateTrainingVariations}
            disabled={loading}
            className="w-full">
            {loading ? <div className="mr-2 h-4 w-4 animate-spin" /> : null}
            Generate Variations
          </button>

          <button onClick={trainNetwork} className="w-full">
            {loading ? <div className="mr-2 h-4 w-4 animate-spin" /> : null}
            Train Network
          </button>

          <button onClick={updateNetwork} disabled={loading} className="w-full">
            {loading ? <div className="mr-2 h-4 w-4 animate-spin" /> : null}
            Update Network
          </button>

          <button
            onClick={generatePalette}
            disabled={loading}
            className="w-full">
            {loading ? <div className="mr-2 h-4 w-4 animate-spin" /> : null}
            Generate Palette
          </button>
        </div>
      </div>
    </div>
  );
};

export default ColorPalette;
