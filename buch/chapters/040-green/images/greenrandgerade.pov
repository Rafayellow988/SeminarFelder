//
// greenrandgerade.pov
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "greenrand.inc"

intersection {
	box { <-1.5,-1,0>, <1.5,3,1.3> }
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

intersection {
	box { <-1.5,-1,-0.1>, <1.5,3,1.3> }
	union {
		randgerade(0.008)
	}
	pigment {
		color kurvefarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

union {
	gittergerade(0.005)
	pigment {
		color gitterfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

mesh {
	wandgerade()
	pigment {
		color wandfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
