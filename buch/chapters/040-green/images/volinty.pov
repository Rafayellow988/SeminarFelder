//
// volintx.pov
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "volintcommon.inc"

union {
	yprismen()
	pigment {
		color prismenfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

intersection {
	cylinder { <0,-1,0>, <0,1,0>, 1.45 }
	union {
		ygrid(0.002)
	}
	pigment {
		color ygridfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

arrow(-1.9 * y1axis, 1.9 * y1axis, 0.01, pfeilfarbe)
arrow(-1.9 * y2axis, 1.9 * y2axis, 0.01, pfeilfarbe)

