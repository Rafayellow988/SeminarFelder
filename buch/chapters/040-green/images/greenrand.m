%
% greenrand.m
%
% (c) 2025 Prof Dr Andreas Müller
%
N = 50;
global lim;
lim = 1.2;
global n;
n = 100;
global xlimits;
xlimits = 1.5;
global ylimit;
ylimit = 1.2;

function retval = m(x)
	global lim;
	y = abs(x) / lim;
	if (y > 1)
		retval = 0;
		return;
	endif
	if (y < 1/2)
		retval = 1;
		return;
	endif
	retval = (cos((y - 1/2)*2*pi) + 1) / 2;
end

function retval = f1(x1)
	retval = (0.7 + 0.4 * x1 - 0.5 * sin(x1 * pi)) * m(x1);
end

function retval = f2(x2)
	global ylimit;
	y = x2 / ylimit;
	if (y > 1)
		retval = 0;
		return;
	endif
	retval = (0.5 + y) * (cos(y * pi) + 1) / 2;
end

function retval = f(x1, x2)
	retval = f1(x1) * f2(x2);
end

function retval = t(p, s)
	retval = p;
	retval(2) = p(2) + s * p(1)^2;
end

function dreieck(fn, p0, p1, p2)
	fprintf(fn, "    triangle {\n");
	fprintf(fn, "        <%.4f,%.4f,%.4f>,\n", p0(1), p0(3), p0(2));
	fprintf(fn, "        <%.4f,%.4f,%.4f>,\n", p1(1), p1(3), p1(2));
	fprintf(fn, "        <%.4f,%.4f,%.4f>\n",  p2(1), p2(3), p2(2));
	fprintf(fn, "    }\n");
end

function flaeche(fn, s)
	global n;
	global xlimits;
	global ylimit;

	xn = n;
	ximin = -xn;
	ximax = xn-1;
	xh = xlimits / xn;

	yn = n;
	yimin = 0;
	yimax = yn-1;
	yh = (ylimit + 0.2) / yn;

	for (xi = (ximin:ximax))
		x = xi * xh;
		for (yi = (yimin:yimax))
			y = yi * yh;
			p0 = [ x,      y,      0 ];
			p1 = [ x + xh, y,      0 ];
			p2 = [ x + xh, y + yh, 0 ];
			p3 = [ x     , y + yh, 0 ];
			p0(3) = f(p0(1), p0(2));
			p1(3) = f(p1(1), p1(2));
			p2(3) = f(p2(1), p2(2));
			p3(3) = f(p3(1), p3(2));
			if (s > 0)
				p0 = t(p0, s);
				p1 = t(p1, s);
				p2 = t(p2, s);
				p3 = t(p3, s);
			endif
			dreieck(fn, p0, p1, p2);
			dreieck(fn, p0, p2, p3);
		end
	end
end

function segment(fn, p0, p1)
	fprintf(fn, "    cylinder {\n");
	fprintf(fn, "        <%.4f,%.4f,%.4f>,\n", p0(1), p0(3), p0(2));
	fprintf(fn, "        <%.4f,%.4f,%.4f>,\n", p1(1), p1(3), p1(2));
	fprintf(fn, "        gr\n");
	fprintf(fn, "    }\n");
	fprintf(fn, "    sphere {\n");
	fprintf(fn, "        <%.4f,%.4f,%.4f>,\n", p1(1), p1(3), p1(2));
	fprintf(fn, "        gr\n");
	fprintf(fn, "    }\n");
end

function kurve(fn, s)
	global n;
	global xlimits;
	global ylimit;

	xn = n;
	ximin = -xn;
	ximax = xn-1;
	xh = xlimits / xn;
	for (xi = (ximin:ximax))
		x = xi * xh;
		p0 = [ x,      0, 0 ]; p0(3) = f(p0(1), p0(2));
		p1 = [ x + xh, 0, 0 ]; p1(3) = f(p1(1), p1(2));
		if (s > 0)
			p0 = t(p0, s);
			p1 = t(p1, s);
		endif
		segment(fn, p0, p1)
	end
end

function gitterliniex(fn, s, x)
	global n;
	global ylimit;

	yn = n;
	yimin = 0;
	yimax = yn-1;
	yh = (ylimit - s * x^2 + 0.1) / yn;

	for (yi = (yimin:yimax))
		y = yi * yh;
		p0 = [ x, y,      0 ];
		p1 = [ x, y + yh, 0 ];
		p0(3) = f(p0(1), p0(2));
		p1(3) = f(p1(1), p1(2));
		if (s > 0)
			p0 = t(p0, s);
			p1 = t(p1, s);
		endif
		segment(fn, p0, p1);
	end
end

function gitterliniey(fn, s, y)
	global n;
	global xlimits;

	xn = n;
	ximin = -xn;
	ximax = xn-1;
	if (s * xlimits^2 > y)
		xh = sqrt(y / s) / xn;
	else
		xh = xlimits / xn;
	endif

	for (xi = (ximin:ximax))
		x = xi * xh;
		p0 = [ x,      y - s *  x^2,       0 ];
		p1 = [ x + xh, y - s * (x + xh)^2, 0 ];
		p0(3) = f(p0(1), p0(2));
		p1(3) = f(p1(1), p1(2));
		if (s > 0)
			p0 = t(p0, s);
			p1 = t(p1, s);
		endif
		segment(fn, p0, p1);
	end
end

function gitter(fn, s)
	for x = (-1.4:0.2:1.4)
		gitterliniex(fn, s, x);
	end
	for y = (0.2:0.2:1.2)
		gitterliniey(fn, s, y);
	end
end

function wand(fn, s)
	global n;
	global xlimits;
	global ylimit;

	xn = n;
	ximin = -xn;
	ximax = xn-1;
	xh = xlimits / xn;
	for (xi = (ximin:ximax))
		x = xi * xh;
		p0 = [ x,      0, 0 ]; p0(3) = f(p0(1), p0(2));
		p1 = [ x + xh, 0, 0 ]; p1(3) = f(p1(1), p1(2));
		if (s > 0)
			p0 = t(p0, s);
			p1 = t(p1, s);
		endif
		p2 = p1; p2(3) = 0;
		p3 = p0; p3(3) = 0;
		dreieck(fn, p0, p1, p2);
		dreieck(fn, p0, p2, p3);
	end
end

fn = fopen("greenranddata.inc", "w");

% Flaeche mit geradem rand
fprintf(fn, "#macro flaechegerade()\n");
flaeche(fn, 0);
fprintf(fn, "#end\n");

% Flaeche mit gekruemmtem rand
fprintf(fn, "#macro flaechegekruemmt()\n");
flaeche(fn, 0.3);
fprintf(fn, "#end\n");

% Randkurve gerade
fprintf(fn, "#macro randgerade(gr)\n");
kurve(fn, 0);
fprintf(fn, "#end\n");

% Randkurve gekruemmt
fprintf(fn, "#macro randgekruemmt(gr)\n");
kurve(fn, 0.3);
fprintf(fn, "#end\n");

% Wand gerade
fprintf(fn, "#macro wandgerade()\n");
wand(fn, 0);
fprintf(fn, "#end\n");

% Wand gekrümmt
fprintf(fn, "#macro wandgekruemmt()\n");
wand(fn, 0.3);
fprintf(fn, "#end\n");

% Gitter gerade
fprintf(fn, "#macro gittergerade(gr)\n");
gitter(fn, 0);
fprintf(fn, "#end\n");

% Gitter gekrümmt
fprintf(fn, "#macro gittergekruemmt(gr)\n");
gitter(fn, 0.3);
fprintf(fn, "#end\n");

fclose(fn);
