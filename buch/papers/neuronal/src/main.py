import os
import boundaries
from WaveNet import WaveNet
from Graphics import *
matplotlib.use('TkAgg')


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveNet().to(device)

if os.path.exists("wavenet.pth"):
    model.load_state_dict(torch.load("wavenet.pth", map_location=device))
    model.eval()
    print("Loaded trained model from file\n")
else:
    print("Training model...")
    # Generate training & testing points
    # Uniformly generated in the x-y-t coordinate system
    n_train_samples = 10000
    x_train = torch.FloatTensor(n_train_samples, 1).uniform_(boundaries.X_MIN, boundaries.X_MAX)
    y_train = torch.FloatTensor(n_train_samples, 1).uniform_(boundaries.Y_MIN, boundaries.Y_MAX)
    t_train = torch.FloatTensor(n_train_samples, 1).uniform_(boundaries.T_MIN, boundaries.T_MAX)

    n_test_samples = 1000
    x_test = torch.FloatTensor(n_test_samples, 1).uniform_(-104, -100)
    y_test = torch.FloatTensor(n_test_samples, 1).uniform_(-104, -100)
    t_test = torch.FloatTensor(n_test_samples, 1).uniform_(0, 5)

    train_error, test_error = model.fit(x_train, y_train, t_train, x_test, y_test, t_test, device, n_epochs=200)
    torch.save(model.state_dict(), "wavenet.pth")
    print("\nModel saved to 'wavenet.pth'\n")

    error_plot(train_error, test_error)

# Animation
#animate_neural_network(model, device)
animate_comparison(model, device)
