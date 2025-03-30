//
// differenz.pov -- Differenz zweier Vektoren entlang einer Kurve
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "diffcommon.inc"

tangentialvektor(T1, pfeilfarbe)
tangentialvektor(T2, pfeilfarbe)

//#declare T = 0;
//pfeiltransportiert(T)

#declare P1 = kurve(T1) + 0.25 * vnormalize(richtung(T1));
#declare P2 = kurve(T2) + 0.25 * vnormalize(richtung(T2));
union {
	cylinder { P1, P2, 0.003 }
	sphere { P1, 0.003 }
	sphere { P2, 0.003 }
	pigment {
		color Red
	}
	finish {
		metallic
		specular 0.9
	}
}

//#declare n = kurve(T1);
//
//#declare west = vnormalize(vcross(n, e2));
//#declare upward = vnormalize(vcross(n, west));
//
//arrow(0, 1.1 * west, 0.01, Blue)
//arrow(0, 1.1 * upward, 0.01, Green)
//
//intersection {
//	plane { n, 0.99 }
//	plane { n, 1.01 }
//	pigment {
//		color rgbf<0.6,0.8,0.6,0.5>
//	}
//	finish {
//		metallic
//	}
//}
