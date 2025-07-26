import os
import boundaries
from WaveNet import WaveNet
from Graphics import *
matplotlib.use('TkAgg')


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveNet().to(device)
print("Model parameters:" + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + "\n")

if os.path.exists("wavenet.pth"):
    model.load_state_dict(torch.load("wavenet.pth", map_location=device))
    model.eval()
    print("Loaded trained model from file\n")
else:
    print("Training model...")
    # Generate training & testing points
    n_samples = 40000
    x = torch.FloatTensor(n_samples, 1).uniform_(boundaries.X_MIN, boundaries.X_MAX)
    y = torch.FloatTensor(n_samples, 1).uniform_(boundaries.Y_MIN, boundaries.Y_MAX)
    t = torch.FloatTensor(n_samples, 1).uniform_(boundaries.T_MIN, boundaries.T_MAX)

    indices = torch.randperm(n_samples)
    split_idx = int(n_samples * 0.8) # 80% train, 20% test
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    t_train = t[train_indices]
    t_test = t[test_indices]

    train_error, test_error = model.fit(x_train, y_train, t_train, x_test, y_test, t_test, device, n_epochs=700)
    torch.save(model.state_dict(), "wavenet.pth")
    print("\nModel saved to 'wavenet.pth'\n")

    error_plot(train_error, test_error)
    model.eval()

# Animation
animate_comparison(model, device)


