//
// moebius.pov -- Moebius band
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

global_settings { ambient_light rgb<2, 2, 2> }

place_camera(<-33, 20, 50>, <-0.05, -0.05, 0>, 16/9, 0.021)
lightsource(<10, 50, 40>, 1, 0.5 * White)
lightsource(<-10, 50, -40>, 1, 0.5 * White)
lightsource(<-10, -10, 10>, 1, 0.5 * White)

#declare l = 0.2;
#declare zylinderfarbe = rgb<0.8,0.8,0.8>;
#declare farbe0 = rgb<1.0,0.6,0.8>;
#declare farbe1 = rgb<0.6,0.8,1.0>;
#declare farbe2 = rgb<0.8,1.0,0.6>;
#declare gridfarbe = rgb<1.0,0.8,0.0>;
#declare gridradius = 0.003;

//arrow(-e1, e1, 0.01, White)
//arrow(-e2, e2, 0.01, White)
//arrow(-e3, e3, 0.01, White)

#macro punkt(phi, phasenwinkel, Y, R)
	R * < cos(phi), 0, R * sin(phi) >
	+
	Y * <	sin((phi + phasenwinkel) / 2) * cos(phi),
		cos((phi + phasenwinkel) / 2),
		sin((phi + phasenwinkel) / 2) * sin(phi) >
#end

#macro karte(winkel,phasenwinkel,farbe,R,shift)
mesh {
	#declare Lmin = -l;
	#declare Lmax = l;
	#declare Lsteps = 20;
	#declare Lstep = (Lmax - Lmin) / Lsteps;
	#declare phimin = -pi / 2;
	#declare phimax =  pi / 2;
	#declare phisteps = 200;
	#declare phistep = (phimax - phimin) / phisteps;
	#declare phi = phimin;
	#while (phi < phimax - phistep/2)
		#declare L = Lmin;
		#while (L < Lmax - Lstep/2)
			#declare p0 = punkt(phi,           phasenwinkel, L,         R);
			#declare p1 = punkt(phi + phistep, phasenwinkel, L,         R);
			#declare p2 = punkt(phi + phistep, phasenwinkel, L + Lstep, R);
			#declare p3 = punkt(phi,           phasenwinkel, L + Lstep, R);
			triangle { p0, p1, p2 }
			triangle { p0, p2, p3 }
			#declare L = L + Lstep;
		#end
		#declare phi = phi + phistep;
	#end
	pigment {
		color farbe
	}
	finish {
		metallic
		specular 0.9
	}
	rotate <0, winkel, 0>
	translate <0, shift, 0>
}
union {
	#declare phimin = -pi / 2;
	#declare phimax =  pi / 2;
	#declare phisteps = 20;
	#declare phistep = (phimax - phimin) / phisteps;
	#declare phi = phimin;
	#while (phi < phimax + phistep/2)
		#declare p0 = punkt(phi, phasenwinkel, -l, R);
		#declare p1 = punkt(phi, phasenwinkel, l, R);
		cylinder { p0, p1, gridradius }
		#declare phi = phi + phistep;
	#end

	#declare L = -l;
	#declare Lstep = l;
	#while (L < l + Lstep/2)
		#declare phisteps = 200;
		#declare phistep = (phimax - phimin) / phisteps;
		#declare phi = phimin;
		#declare p0 = punkt(phi, phasenwinkel, L, R);
		sphere { p0, gridradius }
		#while (phi < phimax - phistep/2)
			#declare phi = phi + phistep;
			#declare p1 = punkt(phi, phasenwinkel, L, R);
			cylinder { p0, p1, gridradius }
			#declare p0 = p1;
			sphere { p0, gridradius }
		#end
		#declare L = L + Lstep;
	#end
	
	pigment {
		color farbe
	}
	finish {
		metallic
		specular 0.9
	}
	rotate <0, winkel, 0>
	translate <0, shift, 0>
}
#end

#declare drehwinkel = 0;

karte(0 + drehwinkel, 0, farbe0, 1.03, 0)
karte(120 + drehwinkel, -radians(120), farbe1, 1, -0.02)
karte(240 + drehwinkel, -radians(240), farbe2, 1,  0.02)
