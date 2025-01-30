%
% volint.m
%
% (c) 2025 Prof Dr Andreas Müller
%
global a;
a = 1;
global n;
n = 100;
global m;
m = 14;

function retval = h(x)
	global	a;
	if (abs(x) > 1)
		retval = 0;
		return;
	endif
	y = (x + 1) / 2;
	retval = a * sin(pi * y)^2;
end

function retval = f(x1, x2)
	retval = h(x1) * h(x2);
end

A = [
   0.8,  0.2;
   0.2,  0.8;
];

function dreieck(fn, p0, p1, p2)
	fprintf(fn, "  triangle {\n");
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p0(1), p0(3), p0(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p1(1), p1(3), p1(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>\n",  p2(1), p2(3), p2(2));
	fprintf(fn, "  }\n");
end

function wand(fn, p, q, w)
	fprintf(fn, "  triangle {\n");
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p(1), 0, p(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", q(1), 0, q(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>\n", q(1), w, q(2));
	fprintf(fn, "  }\n");
	fprintf(fn, "  triangle {\n");
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p(1), 0, p(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", q(1), w, q(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>\n", p(1), w, p(2));
	fprintf(fn, "  }\n");
end

function deckel(fn, p0, p1, p2, p3, w)
	fprintf(fn, "  triangle {\n");
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p0(1), w, p0(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p1(1), w, p1(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>\n", p2(1), w, p2(2));
	fprintf(fn, "  }\n");
	fprintf(fn, "  triangle {\n");
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p0(1), w, p0(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>,\n", p2(1), w, p2(2));
	fprintf(fn, "    <%.4f, %.4f, %.4f>\n", p3(1), w, p3(2));
	fprintf(fn, "  }\n");
end

function quader(fn, p0, p1, p2, p3, w)
	fprintf(fn, "mesh {\n");
	wand(fn, p0, p1, w);
	wand(fn, p1, p2, w);
	wand(fn, p2, p3, w);
	wand(fn, p3, p0, w);
	deckel(fn, p0, p1, p2, p3, 0);
	deckel(fn, p0, p1, p2, p3, w);
	fprintf(fn, "}\n");
end

fn = fopen("volintdata.inc", "w");

fprintf(fn, "#macro flaeche()\n");
x1n = n;
x1h = 2/x1n;
x2n = n;
x2h = 2/x2n;
for x1i = (-x1n:x1n-1)
	x1 = x1i * x1h;
	for x2i = (-x2n:x2n-1)
		x2 = x2i * x2h;
		p0 = [ x1,       x2,       f(x1,       x2)       ];
		p1 = [ x1 + x1h, x2,       f(x1 + x1h, x2)       ];
		p2 = [ x1 + x1h, x2 + x2h, f(x1 + x1h, x2 + x2h) ];
		p3 = [ x1,       x2 + x2h, f(x1,       x2 + x2h) ];
		dreieck(fn, p0, p1, p2);
		dreieck(fn, p0, p2, p3);
	endfor
endfor
fprintf(fn, "#end\n");

fprintf(fn, "#macro xprismen()\n");
x1n = m;
x1h = 2/x1n;
x1d = 0.01 * x1h;
x1u = x1h - x1d;
x2n = m;
x2h = 2/x2n;
x2d = 0.01 * x2h;
x2u = x2h - x2d;
for x1i = (-x1n:x1n-1)
	x1 = x1i * x1h;
	for x2i = (-x2n:x2n-1)
		x2 = x2i * x2h;
		p0 = [ x1 + x1d, x2 + x2d ];
		p1 = [ x1 + x1u, x2 + x2d ];
		p2 = [ x1 + x1u, x2 + x2u ];
		p3 = [ x1 + x1d, x2 + x2u ];
		p = (p0 + p1 + p2 + p3)/4;
		w = f(p(1), p(2));
		if (w > 0)
			quader(fn, p0, p1, p2, p3, w);
		endif
	endfor
endfor
fprintf(fn, "#end\n");

fprintf(fn, "#macro xgrid(gridradius)\n");
x2min = -2;
x2max = 2;
for x1i = (-x1n:x1n-1)
	x1 = x1i * x1h;
	p0 = [ x1, x2min ];
	p1 = [ x1, x2max ];
	fprintf(fn, "cylinder { <%.4f, 0, %.4f>, <%.4f, 0, %.4f>, gridradius }\n", p0(1), p0(2), p1(1), p1(2));
endfor
x1min = -2;
x1max = 2;
for x2i = (-x2n:x2n-1)
	x2 = x2i * x2h;
	p0 = [ x1min, x2 ];
	p1 = [ x1max, x2 ];
	fprintf(fn, "cylinder { <%.4f, 0, %.4f>, <%.4f, 0, %.4f>, gridradius }\n", p0(1), p0(2), p1(1), p1(2));
endfor
fprintf(fn, "#end\n");

fprintf(fn, "#macro yprismen()\n");
y1n = m;
y1h = 2/y1n;
y1d = 0.01 * y1h;
y1u = y1h - y1d;
y2n = m;
y2h = 2/y2n;
y2d = 0.01 * y2h;
y2u = y2h - y2d;
for y1i = (-y1n:y1n-1)
	y1 = y1i * y1h;
	for y2i = (-y2n:y2n-1)
		y2 = y2i * y2h;
		p0 = [ y1 + y1d, y2 + y2d ] * A';
		p1 = [ y1 + y1u, y2 + y2d ] * A';
		p2 = [ y1 + y1u, y2 + y2u ] * A';
		p3 = [ y1 + y1d, y2 + y2u ] * A';
		p = (p0 + p1 + p2 + p3)/4;
		w = f(p(1), p(2));
		if ( w > 0)
			quader(fn, p0, p1, p2, p3, w);
		endif
	endfor
endfor
fprintf(fn, "#end\n");

fprintf(fn, "#macro ygrid(gridradius)\n");
y2min = -2;
y2max = 2;
for y1i = (-y1n:y1n-1)
	y1 = y1i * y1h;
	p0 = [ y1, y2min ] * A';
	p1 = [ y1, y2max ] * A';
	fprintf(fn, "cylinder { <%.4f, 0, %.4f>, <%.4f, 0, %.4f>, gridradius }\n", p0(1), p0(2), p1(1), p1(2));
endfor
y1min = -2;
y1max = 2;
for y2i = (-y2n:y2n-1)
	y2 = y2i * y2h;
	p0 = [ y1min, y2 ] * A';
	p1 = [ y1max, y2 ] * A';
	fprintf(fn, "cylinder { <%.4f, 0, %.4f>, <%.4f, 0, %.4f>, gridradius }\n", p0(1), p0(2), p1(1), p1(2));
endfor
fprintf(fn, "#end\n");

fprintf(fn, "#declare y1axis = <%.4f, 0, %.4f>;\n", A(1,1), A(2,1));
fprintf(fn, "#declare y2axis = <%.4f, 0, %.4f>;\n", A(1,2), A(2,2));

fclose(fn);

