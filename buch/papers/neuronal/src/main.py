from WaveNet import WaveNet
from Graphics import *
matplotlib.use('TkAgg')


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveNet().to(device)

# Generate training & testing points
# Uniformly generated in the x-y-t coordinate system
n_train_samples = 5000
n_test_samples = 1000
x_train = torch.FloatTensor(n_train_samples, 1).uniform_(-2, 2)
y_train = torch.FloatTensor(n_train_samples, 1).uniform_(-2, 2)
t_train = torch.FloatTensor(n_train_samples, 1).uniform_(0, 5)
xyt_train = torch.cat([x_train, y_train, t_train], dim=1).to(device)

x_test = torch.FloatTensor(n_test_samples, 1).uniform_(-104, -100)
y_test = torch.FloatTensor(n_test_samples, 1).uniform_(-104, -100)
t_test = torch.FloatTensor(n_test_samples, 1).uniform_(0, 5)
xyt_test = torch.cat([x_test, y_test, t_test], dim=1).to(device)

# Training
train_error, test_error = model.fit(xyt_train, xyt_test, 200)

# Plots
#plot_solution(model, device, t_fixed=0.5)
animate_solution(model, device)
#error_plot(train_error, test_error)
