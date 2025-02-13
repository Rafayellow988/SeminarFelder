//
// oberflaeche.pov -- 
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<23, 20, 50>, <0.5, 0.688, 0.5>, 16/9, 0.0305)
lightsource(<40, 25, 20>, 1, White)

#declare wuerfelfarbe = rgb<0.8,0.8,0.8>;

arrow(-0.1 * e1, 1.2 * e1, 0.01, White)
arrow(-0.1 * e2, 1.2 * e2, 0.01, White)
arrow(-0.1 * e3, 1.2 * e3, 0.01, White)

box { <0, 0, 0>, <1, 1, 1>
	pigment {
		color wuerfelfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#declare farbe1 = rgb<1.0,0.8,0.8>;
#declare farbe2 = rgb<0.8,1.0,0.8>;
#declare farbe3 = rgb<0.8,0.8,1.0>;
#declare vektorfarbe = rgb<0.8,0.8,0.0>;

#declare d = 0.25;

arrow(< d, 1.0, d >, < 1-d, 1.0,     d >, 0.014, farbe1)
arrow(< d, 1.0, d >, <   d, 1.0,   1-d >, 0.014, farbe2)
arrow(< d, 1.0, d >, <   d, 2-2*d,   d >, 0.014, farbe3)

arrow(<1.0,d,d>, <1.0,d,1-d>, 0.014, farbe2)
arrow(<1.0,d,d>, <1.0,1-d,d>, 0.014, farbe3)
arrow(<1.0,d,d>, <2-2*d,d,d>, 0.014, farbe1)

arrow(< d, d, 1.0 >, <   d, 1-d, 1.0   >, 0.014, farbe3)
arrow(< d, d, 1.0 >, < 1-d,   d, 1.0   >, 0.014, farbe1)
arrow(< d, d, 1.0 >, <   d,   d, 2-2*d >, 0.014, farbe2)

union {
	box { <     d, 0.999, d     >, <   1-d, 1.001, 1-d   > }
	box { < 0.999,     d, d     >, < 1.001,   1-d, 1-d   > }
	box { <     d,     d, 0.999 >, <   1-d,   1-d, 1.001 > }
	pigment {
		color vektorfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
