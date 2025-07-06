% Definieren der Funktion
f = @(x, y) x.^2 + y.^2;

% Bereich der x- und y-Werte
x = linspace(-5, 5, 100);
y = linspace(-5, 5, 100);
[X, Y] = meshgrid(x, y);
Z = f(X, Y);

% Erstellen eines 3D-Oberflächendiagramms
figure;
surf(X, Y, Z);
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
title('3D-Oberfläche für f(x, y) = x^2 + y^2');
shading interp;
colorbar;
pause;
