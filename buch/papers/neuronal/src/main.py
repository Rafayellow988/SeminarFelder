from WaveNet import WaveNet
from Graphics import *
matplotlib.use('TkAgg')


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveNet().to(device)

# Generate training & testing points
# Uniformly generated in the x-y-t coordinate system
n_train_samples = 5000

x_train = torch.FloatTensor(n_train_samples, 1).uniform_(-2, 2)
y_train = torch.FloatTensor(n_train_samples, 1).uniform_(-2, 2)
t_train = torch.FloatTensor(n_train_samples, 1).uniform_(0, 5)

n_test_samples = 1000

x_test = torch.FloatTensor(n_test_samples, 1).uniform_(-104, -100)
y_test = torch.FloatTensor(n_test_samples, 1).uniform_(-104, -100)
t_test = torch.FloatTensor(n_test_samples, 1).uniform_(0, 5)

# Training
train_error, test_error = model.fit(x_train, y_train, t_train, x_test, y_test, t_test, device, 200)

# Plots
#plot_solution(model, device, t_fixed=0.5)
animate_solution(model, device)
#error_plot(train_error, test_error)
