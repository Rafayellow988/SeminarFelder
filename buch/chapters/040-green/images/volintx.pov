//
// volintx.pov
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "volintcommon.inc"

union {
	xprismen()
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
		xgrid(0.002)
	}
	pigment {
		color gridfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
