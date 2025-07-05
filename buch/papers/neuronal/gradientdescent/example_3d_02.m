% Definieren der Funktion
f = @(x, y) sin(sqrt(x.^2 + y.^2));

% Bereich der x- und y-Werte
x = linspace(-10, 10, 100);
y = linspace(-10, 10, 100);
[X, Y] = meshgrid(x, y);
Z = f(X, Y);

% Erstellen eines 3D-Drahtgitterdiagramms
figure;
mesh(X, Y, Z);
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
title('3D-Drahtgitter für f(x, y) = sin(sqrt(x^2 + y^2))');
colorbar;
pause;
