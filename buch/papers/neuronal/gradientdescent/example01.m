% Definieren der Funktion
f = @(x) (x - 3).^2;

% Bereich der x-Werte
x = linspace(0, 6, 400);
y = f(x);

% Plotten der Funktion
figure;
plot(x, y, 'b-', 'LineWidth', 2);
hold on;
grid on;
xlabel('x');
ylabel('f(x)');
title('Quadratische Funktion f(x) = (x - 3)^2');
legend('f(x)', 'Location', 'northeast');
hold off;
pause;
