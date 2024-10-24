import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class PaletteDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor([item["input"]["r"], item["input"]["g"], item["input"]["b"]], dtype=torch.float32)

        colors = []
        for i in range(10):
            color = item["output"][f"color{i+1}"]
            colors.extend([color["r"], color["g"], color["b"]])

        y = torch.tensor(colors, dtype=torch.float32)
        return x, y

class PaletteNet(nn.Module):
    def __init__(self, num_colors=10):
        super(PaletteNet, self).__init__()
        self.num_colors = num_colors
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_colors * 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
class ColorPaletteNetwork:
    def __init__(self):
        self.model = PaletteNet()
        self.load_model()

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load("palette_model.pth"))
            self.model.eval()
        except:
            print("No existing model for color palettes found")

    def save_palette(self, input_rgb, output_palette, is_generated=False):
        try:
            with open("palette_data.json", "r") as f:
                data = json.load(f)
            
            new_entry = {
                "input": {
                    "r": float(input_rgb[0]),
                    "g": float(input_rgb[1]),
                    "b": float(input_rgb[2])
                },
                "output": {
                    f"color{i+1}": {
                        "r": float(color[0]),
                        "g": float(color[1]),
                        "b": float(color[2])
                    } for i, color in enumerate(output_palette)
                },
                "metadata": {
                    "is_generated": is_generated,
                    "version": "1.0"
                }
            }

            data.append(new_entry)

            with open("palette_data.json", "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"failed to save palette: {e}")
            return False
    
    def add_training_data(self, input_rgb, output_palette):
        if len(output_palette) != 10:
            raise ValueError("OutputPalette mus contain exactly 10 colors")
        
        return self.save_palette(input_rgb, output_palette, is_generated=False)

    def train(self):
        batch_size = 32
        num_epochs = 100
        learning_rate = 0.001

        dataset = PaletteDataset("palette_data.json")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), "palette_model.pth")
        return {"message": "Palette training completed", "final_loss": avg_loss}
    
    def generate_palette(self, rgb_values):
        """
        Generates palette with 10 variants
        """

        if not all(0 <= x <= 1 for x in rgb_values) or len(rgb_values) != 3:
            raise ValueError("RGB values must be between 0 and 1")
        test_input = torch.tensor(rgb_values, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.model(test_input)
            palette = prediction.numpy().reshape(-1, 3)
            return palette.tolist()
        
    def generate_training_variations(self, base_rgb):
        """
        Helper for generation of training data
        Generates exactly 10 color variations:
        - Base color
        - 5 lighter variations
        - 4 darker variations
        """
        variations = []
        variations.append(base_rgb)  # Base color
    
        # 5 lighter variations
        for i in range(1, 6):  # Changed from range(1, 5) to range(1, 6)
            factor = 1 + (i * 0.1)
            variation = [min(1.0, c * factor) for c in base_rgb]
            variations.append(variation)
    
        # 4 darker variations
        for i in range(1, 5):
            factor = 1 - (i * 0.1)
            variation = [max(0.0, c * factor) for c in base_rgb]
            variations.append(variation)
    
        return variations