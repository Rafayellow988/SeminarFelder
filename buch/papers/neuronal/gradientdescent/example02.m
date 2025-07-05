% Definieren der Funktion und ihrer Ableitung
f = @(x) (x - 3).^2;
f_prime = @(x) 2*(x - 3);

% Initialisierung
x = 0;
learning_rate = 0.1;
num_iterations = 20;

% Arrays zum Speichern der Werte
x_history = zeros(1, num_iterations + 1);
f_history = zeros(1, num_iterations + 1);
x_history(1) = x;
f_history(1) = f(x);

% Gradientenabstieg
for i = 1:num_iterations
    x = x - learning_rate * f_prime(x);
    x_history(i + 1) = x;
    f_history(i + 1) = f(x);
end

% Plotten der Funktion und des Abstiegsverlaufs
figure;
plot(x_history, f_history, 'ro-', 'LineWidth', 2, 'MarkerSize', 5);
hold on;
f_plot = f(x);
x_plot = linspace(0, 6, 400);
y_plot = f(x_plot);
plot(x_plot, y_plot, 'b-', 'LineWidth', 1);
hold off;
grid on;
xlabel('x');
ylabel('f(x)');
title('Gradientenabstieg für f(x) = (x - 3)^2');
legend('Abstiegspfad', 'f(x)', 'Location', 'northeast');
pause;
