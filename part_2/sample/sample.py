import torch
import torch.nn as nn
import torch.optim as optim


# 1. Dataset definition
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

print(x_train.shape)

# 2. Model definition
model = nn.Linear(1,1)

# 3. Loss function and optimizer settings
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 4. Model training
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# 5. Model evaluation
test_x = torch.tensor([[1.5],[2.5],[3.5],[4.5]])
model.eval()
with torch.no_grad():
    predicted = model(test_x)
    print(predicted)

# 6. save check point
torch.save(model.state_dict(), "./check_point_sample.pt")


# 7. load check point
model.load_state_dict(torch.load("./check_point_sample.pt"))
model.eval()

load_test_x = torch.tensor([[50,100], [150,200], [250, 300]], dtype = torch.float32).reshape([-1,1])
print(model(load_test_x))
