//
// band.pov -- Karten eines zylindrischen Bandes
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

global_settings { ambient_light rgb<2, 2, 2> }

place_camera(<-33, 20, 50>, <0, 0, 0>, 16/9, 0.02)
lightsource(<10, 50, 40>, 1, White)

#declare l = 0.2;
#declare zylinderfarbe = rgb<0.8,0.8,0.8>;
#declare farbe0 = rgb<1.0,0.6,0.8>;
#declare farbe1 = rgb<0.6,0.8,1.0>;
#declare farbe2 = rgb<0.8,1.0,0.6>;

difference {
	cylinder { <0,-l,0>, <0,l,0>, 1 }
	cylinder { <0,-2*l,0>, <0,2*l,0>, 0.99 }
	pigment {
		color zylinderfarbe
	}
	finish {
		metallic
		specular 0.9
	}
	translate <0, 0.0, 0>
}

#declare r = function(phi) { 1.06 + 0.02 * phi }

#macro punkt(phi, Y)
	< r(phi) * cos(phi), Y, r(phi) * sin(phi) >
#end

#macro karte(winkel,farbe)
mesh {
	#declare phimin =    -pi / 6;
	#declare phimax = 5 * pi / 6;
	#declare phistep = (phimax - phimin) / 100;
	#declare phi = phimin;
	#while (phi < phimax - phistep/2)
		#declare p0 = punkt(phi, -l);
		#declare p1 = punkt(phi + phistep, -l);
		#declare p2 = punkt(phi + phistep,  l);
		#declare p3 = punkt(phi,  l);
		triangle { p0, p1, p2 }
		triangle { p0, p2, p3 }
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
}
#end

#declare drehwinkel = 70;

karte(0 + drehwinkel, farbe0)
karte(120 + drehwinkel, farbe1)
karte(240 + drehwinkel, farbe2)
