//
// kartkugel.pov -- Kugelkoordinaten und karteische Koordinaten
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

#declare r = 0.9;
#declare phi = 67 * 3.14159 / 180;
#declare theta = 57 * 3.14159 / 180;

#declare kugelcolor = rgb<1.0,0.4,0.4>;	// rot
#declare kugelcolorf = rgbt<1.0,0.4,0.4,0.5>;	// rot
#declare kartcolor = rgb<0.4,0.6,1.0>;	// blau
#declare kartcolorf = rgb<0.4,0.6,1.0,0.7>;	// blau
#declare punktcolor = rgb<0.2,0.8,0.0>;	// gruen

place_camera(<50, 20, 33>, <0, 0.32, 0>, 16/9, 0.021)
lightsource(<40, 20, 10>, 1, White)

arrow(-e1, e1, 0.01, White)
arrow(-e2, e2, 0.01, White)
arrow(-e3, e3, 0.01, White)

#declare O = <0, 0, 0>;
#declare rf = r * sin(theta);
#declare P  = < rf * cos(phi), r * cos(theta), rf * sin(phi) >;
#declare Pf = < rf * cos(phi), 0,              rf * sin(phi) >;

union {
	sphere { Pf, 0.02 }
	cylinder { Pf, <Pf.x, 0, 0>, 0.01 }
	cylinder { Pf, <0, 0, Pf.z>, 0.01 }
	cylinder { Pf, P, 0.01 }
	pigment {
		color kartcolor
	}
}

intersection {
	cylinder { <0, -0.0025, 0>, <0, 0.0025, 0>, rf }
	plane { <0, 0, -1>, 0 }
	plane { < -sin(phi), 0, cos(phi) >, 0}
	pigment {
		color kugelcolorf
	}
	no_shadow
}

cylinder { O, P, 0.01
	pigment {
		color kugelcolor
	}
	finish {
		metallic
		specular 0.99
	}
}

#declare achse = < -sin(phi), 0, cos(phi) >;

intersection {
	cylinder { -0.0025 * achse, 0.0025 * achse, r }
	plane { < cos(theta) * cos(phi), -sin(theta), cos(theta) * sin(phi) >, 0 }
	plane { < -cos(phi), 0, -sin(phi) >, 0 }
	pigment {
		color kugelcolorf
	}
	no_shadow
}

union {
	sphere { P, 0.02 }
	pigment {
		color punktcolor
	}
	finish {
		metallic
		specular 0.99
	}
}

