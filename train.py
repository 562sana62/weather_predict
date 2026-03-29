import torch
from torch.utils.data import DataLoader,TensorDataset
from models.lstm_model import WeatherLSTM
from preprocess import load_data, create_sequences

data,scaler = load_data("data/hyderabad_weather.csv")

X,y = create_sequences(data)

X = torch.tensor(X).float()
y = torch.tensor(y.reshape(len(y),-1)).float()

dataset = TensorDataset(X,y)

loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = WeatherLSTM(input_size=4,
                    hidden_size=64,
                    num_layers=2)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(50):

    for batch_x,batch_y in loader:
        pred = model(batch_x)
        loss = criterion(pred,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:",epoch,"Loss:",loss.item())

torch.save(model.state_dict(),"weather_model.pth")