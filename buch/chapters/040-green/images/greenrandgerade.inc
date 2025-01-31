//
// greenrandgerade.inc
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "greenrand.inc"

intersection {
	cylinder { <0,-1,0>, <0,3,0>, 1.45 }
	mesh {
		flaechegerade()
	}
	pigment {
		color flaechenfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}




