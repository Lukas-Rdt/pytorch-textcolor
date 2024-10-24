import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RGBDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor([item["r"], item["g"], item["b"]], dtype=torch.float32)
        y = torch.tensor([item["output"]], dtype=torch.float32)
        return x, y
    
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class NeuralNetwork:
    def __init__(self):
        self.model = SimpleNet()
        self.load_model()

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load("rgb_model.pth"))
            self.model.eval()
        except:
            print("No existing model found, use new one")

    def add_training_data(self, rgb_values, output):
        try:
            with open("data.json", "r") as f:
                data = json.load(f)

            new_entry = {
                "r": rgb_values[0],
                "g": rgb_values[1],
                "b": rgb_values[2],
                "output": output
            }
            data.append(new_entry)

            with open("data.json", "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to add new training entry: {e}")
            return False

    def train(self):
        batch_size = 32
        num_epochs = 100
        learning_rate = 0.001

        dataset = RGBDataset("data.json")
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4}")

        torch.save(self.model.state_dict(), "rgb_model.pth")
        return {"message": "Training completed", "final_loss": avg_loss}
    
    def predict(self, rgb_values):
        if not all(0 <= x <= 1 for x in rgb_values) or len(rgb_values) != 3:
            raise ValueError("RGB values must be between 0 and 1")
        test_input = torch.tensor(rgb_values, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.model(test_input)
            return float(prediction.item())