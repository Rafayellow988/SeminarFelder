//
// kurve.pov -- Kurve auf einer Kugeloberfläche
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

#declare kugelfarbe = rgb<0.8,0.8,0.8>;
#declare darkred = rgb<0.8,0.2,0.2>;
#declare darkblue = rgb<0.2,0.4,1.0>;
#declare darkgreen = rgb<0.2,0.6,0.4>;

place_camera(<53, 20, 20>, <0, 0.33, 0>, 16/9, 0.0187)
lightsource(<10, 50, -5>, 1, White)

/*
arrow(-1.2 * e1, 1.2 * e1, 0.01, White)
arrow(-1.2 * e2, 1.2 * e2, 0.01, White)
arrow(-1.2 * e3, 1.2 * e3, 0.01, White)
*/

#declare fromwinkel = radians(-130);
#declare towinkel = radians(-180);

intersection {
	plane { < 0, -1, 0>, -0.1 }
	plane { < cos(fromwinkel), 0, sin(fromwinkel) >, 0 }
	plane { < cos(towinkel), 0, sin(towinkel) >, 0 }
	difference {
		sphere { <0, 0, 0>, 1 }
		sphere { <0, 0, 0>, 0.999 }
		cylinder { <0, -1.5, 0>, <0, 1.5, 0>, 0.3 }
	}
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.50
	}
}

#declare X0 = function(T) { T }
#declare Y0 = function(T) { 0.3 * T }

#declare g1x = function(T) {  0.1 * T * T }
#declare g1y = function(T) {  0.2 * T * T }
#declare g2x = function(T) {  0.1 * T * T * T - 0.1 * T * T }
#declare g2y = function(T) { -0.2 * T * T }

#declare theta = function(Y) { radians(55 - 30 * Y) }
#declare phi = function(X) { radians(30 + 30 * X) }

#macro kugelpunkt(X, Y)
	< sin(theta(Y)) * cos(phi(X)), cos(theta(Y)), sin(theta(Y)) * sin(phi(X)) >
#end

#macro punkt1(T)
	kugelpunkt(X0(T) + g1x(T), Y0(T) + g1y(T))
#end

#macro punkt2(T)
	kugelpunkt(X0(T) + g2x(T), Y0(T) + g2y(T))
#end

#declare kurveradius = 0.012;

union {
	#declare minT = -1.5;
	#declare maxT = 1.5;
	#declare Tsteps = 50;
	#declare Tstep = (maxT - minT) / Tsteps;
	#declare T = minT;
	sphere { punkt1(T), kurveradius }
	#while (T < maxT - Tstep/2)
		cylinder { punkt1(T), punkt1(T + Tstep), kurveradius }
		#declare T = T + Tstep;
		sphere { punkt1(T), kurveradius }
	#end
	pigment {
		color darkred
	}
	finish {
		metallic
		specular 0.99
	}
}

union {
	#declare minT = -1.5;
	#declare maxT = 1.5;
	#declare Tsteps = 50;
	#declare Tstep = (maxT - minT) / Tsteps;
	#declare T = minT;
	sphere { punkt2(T), kurveradius }
	#while (T < maxT - Tstep/2)
		cylinder { punkt2(T), punkt2(T + Tstep), kurveradius }
		#declare T = T + Tstep;
		sphere { punkt2(T), kurveradius }
	#end
	pigment {
		color darkblue
	}
	finish {
		metallic
		specular 0.99
	}
}

#declare tangente = vnormalize(punkt1(0.0001) - punkt1(0));

arrow(punkt1(0), punkt1(0) + 0.5 * tangente, kurveradius, darkgreen)

sphere { punkt1(0), 1.5 * kurveradius
	pigment {
		color rgb<0.8,0.2,1.0>
	}
	finish {
		metallic
		specular 0.99
	}
}
