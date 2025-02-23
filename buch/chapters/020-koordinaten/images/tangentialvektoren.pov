//
// tangentialvektoren.pov -- wo liegen die Tangentialvektoren
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<33, 20, 50>, <0, 0.31, 0>, 16/9, 0.03)
lightsource(<60, 50, 40>, 1, White)

arrow(-e1, 1.3 * e1, 0.012, White)
arrow(-e2, 1.3 * e2, 0.012, White)
arrow(-e3, 1.3 * e3, 0.012, White)

#declare O = <0, 0, 0>;
#declare kugelfarbe = rgb<0.8,0.8,0.8>;
#declare phifarbe = rgb<1.0,0.6,0.8>;
#declare thetafarbe = rgb<0.6,0.8,1.0>;
#declare pfeilfarbe = rgb<0.6,1.0,0.8>;
#declare gitterradius = 0.01;

#declare arrowradius = 0.007;
#declare l = 0.45;

sphere { O, 1
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#macro kugel(theta, phi)
	<
		sin(theta) * cos(phi),
		cos(theta),
		sin(theta) * sin(phi)
	>
#end


#macro breitenkreis(theta)
	#declare phimin = 0;
	#declare phimax = 2 * pi;
	#declare phisteps = 200;
	#declare phistep = (phimax - phimin) / phisteps;
	#declare phi = phimin;
	#while (phi < phimax - phistep/2)
		sphere { kugel(theta, phi), gitterradius }
		cylinder {
			kugel(theta, phi),
			kugel(theta, phi + phistep),
			gitterradius
		}
		#declare phi = phi + phistep;
	#end
#end

// Koordinatengitter
union {
	#declare theta = radians(165);
	#declare thetastep = radians(15);
	#while (theta > thetastep/2)
		breitenkreis(theta)
		#declare theta = theta - thetastep;
	#end
	pigment {
		color phifarbe
	}
	finish {
		metallic
		specular 0.99
	}
}

#macro laengenkreis(phi)
	#declare thetamin = 0;
	#declare thetamax = radians(165);
	#declare thetastep = radians(180) / 100;
	#declare theta = thetamin;
	sphere { kugel(theta, phi), gitterradius }
	#while (theta < thetamax - thetastep/2)
		cylinder {
			kugel(theta, phi),
			kugel(theta + thetastep, phi),
			gitterradius
		}
		#declare theta = theta + thetastep;
		sphere { kugel(theta, phi), gitterradius }
	#end
#end

union {
	#declare phimax = radians(360);
	#declare phistep = radians(15);
	#declare phi = 0;
	#while (phi < phimax - phistep / 2)
		laengenkreis(phi)
		#declare phi = phi + phistep;
	#end
	pigment {
		color thetafarbe
	}
	finish {
		metallic
		specular 0.99
	}
}

#include "tangentialvektoren.inc"

#macro pfeil(wo, richtung)
	arrow(wo, wo + l * richtung, arrowradius, pfeilfarbe)
#end

union {
	pfeile()
	pigment {
		color pfeilfarbe
	}
	finish {
		metallic
		specular 0.99
	}
}
