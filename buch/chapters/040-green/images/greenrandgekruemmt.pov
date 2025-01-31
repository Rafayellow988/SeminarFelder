//
// greenrandgekruemmt.pov
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "greenrand.inc"

intersection {
	box { <-1.5,-1,0>, <1.5,3,1.3> }
	mesh {
		flaechegekruemmt()
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
		randgekruemmt(0.008)
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
	gittergekruemmt(0.005)
	pigment {
		color gitterfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

mesh {
	wandgekruemmt()
	pigment {
		color wandfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
