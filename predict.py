import torch
import numpy as np
from models.lstm_model import WeatherLSTM

model = WeatherLSTM(4,64,2)
model.load_state_dict(torch.load("weather_model.pth"))

model.eval()

last_30_days = np.random.rand(1,30,4)

input_tensor = torch.tensor(last_30_days).float()

prediction = model(input_tensor)

prediction = prediction.detach().numpy()

prediction = prediction.reshape(5,2)

print("5 Day Forecast")

for i in range(5):

    print("Day",i+1)

    print("Temp:",prediction[i][0])
    print("Rainfall:",prediction[i][1])