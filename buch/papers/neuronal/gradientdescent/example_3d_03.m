% Definieren der Funktion
f = @(x, y) (sin(x) .* cos(y)) ./ (x.^2 + y.^2 + 1);

% Bereich der x- und y-Werte
x = linspace(-2*pi, 2*pi, 100);
y = linspace(-2*pi, 2*pi, 100);
[X, Y] = meshgrid(x, y);
Z = f(X, Y);

% Erstellen eines 3D-Oberflächendiagramms
figure;
surf(X, Y, Z);
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
title('3D-Oberfläche für f(x, y) = (sin(x)cos(y))/(x^2 + y^2 + 1)');
shading interp;
colorbar;
pause;
