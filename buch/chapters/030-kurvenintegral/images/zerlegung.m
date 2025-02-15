%
% zerlegung.m
%
% (c) 2025 Prof Dr Andreas Müller
%
global N;
N = 1200;
global n;
n = 6;

function retval = phi(a, b, x)
	if ((x < a) || (x > b))
		retval = 0;
		return;
	end
	t = 2 * pi * (x - a) / (b - a);
	retval = (1 - cos(t)) / 2;
end

global x;
x = linspace(0, 12, 1200)

psi = zeros(n, N);

function retval = einfuellen(psi, a, b, k)
	global N;
	global x;
	for i = (1:N)
		x0 = x(i);
		psi(k, i) = phi(a, b, x0);
	end
	retval = psi;
end

psi = einfuellen(psi, -3, 3, 1);
psi = einfuellen(psi, 9, 15, 2);

fn = fopen("zerlegungpath.tex", "w");

fprintf(fn, "\\def\\aone{-3}\n");
fprintf(fn, "\\def\\bone{3}\n");
fprintf(fn, "\\def\\atwo{9}\n");
fprintf(fn, "\\def\\btwo{15}\n");

psi = einfuellen(psi, 2, 6, 3);
fprintf(fn, "\\def\\athree{2}\n");
fprintf(fn, "\\def\\bthree{6}\n");
psi = einfuellen(psi, 3, 9, 4);
fprintf(fn, "\\def\\afour{3}\n");
fprintf(fn, "\\def\\bfour{9}\n");
psi = einfuellen(psi, 5, 10, 5);
fprintf(fn, "\\def\\afive{5}\n");
fprintf(fn, "\\def\\bfive{10}\n");
psi = einfuellen(psi, 8, 11, 6);
fprintf(fn, "\\def\\asix{8}\n");
fprintf(fn, "\\def\\bsix{11}\n");

normierung = sum(psi, 1)

for i = (1:n)
	psi(i,:) = psi(i,:) ./ normierung;
end

psi


function weg(fn, psi, name, k)
	global	x;
	global	N;
	
	fprintf(fn, "\\def\\%s{\n\t({%.4f*\\dx},{%0.4f*\\dy})", name,
		x(1), psi(k, 1));
	for i = (2:N)
		fprintf(fn, "\n\t-- ({%.4f*\\dx},{%0.4f*\\dy})",
			x(i), psi(k, i));
	end
	fprintf(fn, "\n}\n");
end

weg(fn, psi, "psione", 1);
weg(fn, psi, "psitwo", 2);
weg(fn, psi, "psithree", 3);
weg(fn, psi, "psifour", 4);
weg(fn, psi, "psifive", 5);
weg(fn, psi, "psisix", 6);

cpsi = zeros(n, N);
cpsi(1,:) = psi(1,:);
for i = (2:n)
	cpsi(i,:) = cpsi(i-1,:) + psi(i,:);
end

cpsi

weg(fn, cpsi, "cpsione", 1);
weg(fn, cpsi, "cpsitwo", 2);
weg(fn, cpsi, "cpsithree", 3);
weg(fn, cpsi, "cpsifour", 4);
weg(fn, cpsi, "cpsifive", 5);
weg(fn, cpsi, "cpsisix", 6);

