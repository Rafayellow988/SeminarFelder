//
// stereographisch.pov -- Stereographische Projekt, 3D-Bild
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

#declare O = <0, 0, 0>;

#declare punktcolor = rgb<1,0.8,0>;
#declare meridiancolor = rgb<1, 0.4, 2>;
#declare koordinatenfarbe = rgb<0.8,0,0>;
#declare ebenenfarbe = rgbt<0.4,1.0,0.8,0.5>;

place_camera(<50, 20, 33>, <0, 0, 0>, 16/9, 0.037)
lightsource(<40, 45, -20>, 1, White)

#macro kugel(phi, theta)
	< sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi) >
#end

#declare r = 2.2;
#declare kr = 0.04;
#declare lr = 0.02;
#declare N = < 0, 1, 0>;
#declare phi0 = radians(60);
#declare P1 = < r * cos(phi0), 0, r * sin(phi0) >;
#declare w = pi - 2 * atan(r);

arrow(-1.2 * e1, 1.8 * e1, lr, White)
arrow(-1.2 * e2, 1.2 * e2, lr, White)
arrow(-1.2 * e3, 2.2 * e3, lr, White)

sphere { O, 1
	pigment {
		color rgbt<0.6,0.8,1,0.5>
	}
	finish {
		metallic
		specular 0.99
	}
	no_shadow
}

union {
	sphere { P1, kr }
	sphere { N, kr }
	cylinder { N, P1, lr }
	sphere { kugel(phi0, w), kr }
	pigment {
		color punktcolor
	}
	finish {
		metallic
		specular 0.99
	}
}

union {
	#declare theta = 0;
	#declare thetastep = pi / 60;
	#while (theta < pi - thetastep/2)
		sphere { kugel(phi0, theta), lr }
		cylinder {	kugel(phi0, theta),
				kugel(phi0, theta + thetastep), lr }
		#declare theta = theta + thetastep;
	#end
	pigment {
		color meridiancolor
	}
	finish {
		metallic
		specular 0.99
	}
}

union {
	cylinder { P1, <P1.x, 0, 0>, lr }
	cylinder { P1, < 0, 0, P1.z>, lr }
	pigment {
		color koordinatenfarbe
	}
	finish {
		metallic
		specular 0.99
	}
}

intersection {
	box { <-1.1, -0.1, -1.1>, <1.7, 0.1, 2.1> }
	plane { <0, 1, 0>, 0.005 }
	plane { <0, -1, 0>, 0.005 }
	pigment {
		color ebenenfarbe
	}
	finish {
		metallic
		specular 0.95
	}
}
