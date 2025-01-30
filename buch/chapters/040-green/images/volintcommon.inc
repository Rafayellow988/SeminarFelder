//
// volintcommon.inc
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"
#include "volintdata.inc"

place_camera(<33, 20, 50>, <0, 0.35, 0>, 16/9, 0.026)
lightsource(<40, 15, 10>, 1, White)

arrow(-1.2 * e1, 1.6 * e1, 0.01, White)
arrow(-0.2 * e2, 1.2 * e2, 0.01, White)
arrow(-1.2 * e3, 1.6 * e3, 0.01, White)

#declare prismenfarbe = rgb<1.0,0.6,0.2>;
#declare flaechenfarbe = rgbt<0.6,0.8,0.8,0.4>;
#declare traegerfarbe = rgb<0.6,0.2,1.0>;
#declare gridfarbe = rgb<1.0,0.2,1.0>;
#declare pfeilfarbe = rgb<0.2,0.6,0.4>;
#declare ygridfarbe = rgb<0,0.8,0.4>;

box { <-1,-0.002,-1>, <1,0,1> 
	pigment {
		color traegerfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

intersection {
	cylinder { <0,-1,0>, <0,3,0>, 1.45 }
	mesh {
		flaeche()
	}
	pigment {
		color flaechenfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
