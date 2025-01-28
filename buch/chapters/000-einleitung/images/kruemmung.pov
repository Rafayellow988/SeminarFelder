//
// kruemmung.pov -- Drehung eines Vektors beim Paralleltransport
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<33, 20, 50>, <0, 0.3, 0>, 16/9, 0.024)
lightsource(<30, 20, 30>, 1, White)

arrow(-e1, e1, 0.01, White)
arrow(-e2, e2, 0.01, White)
arrow(-e3, e3, 0.01, White)

sphere { O, 1
	pigment {
		//color rgb<0.8,0.8,0.8>
		image_map {
			png "2k_earth_daymap.png" gamma 1.0
			map_type 1
		}
	}
	finish {
		metallic
		specular 0.9
	}
	rotate <0, 180, 0>
}

#declare punktradius = 0.025;
#declare wegfarbe = rgb<1,0.8,0.2>;

#macro kugel(theta, phi)
	< sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi) >
#end

#declare pfeilradius = 0.02;
#declare pfeillaenge = 0.3;
#declare pfeilfarbe = rgb<0.8,0.2,0.2>;

#macro phitangente(theta, phi)
	arrow(kugel(theta, phi),
		kugel(theta, phi) + pfeillaenge * < -sin(phi), 0, cos(phi) >,
		pfeilradius, pfeilfarbe)
#end

#declare phi = 0;
#declare phimax = pi/2;
#declare phistep = pi / 10;
#while (phi <= phimax)
	phitangente(pi/2, phi)
	#declare phi = phi + phistep;
#end

#declare thetastep = pi / 10;
#declare theta = thetastep;
#declare thetamax = pi/2;
#while (theta < thetamax - thetastep/2)
	arrow(kugel(theta, pi/2),
		kugel(theta, pi/2) + pfeillaenge * <-1, 0, 0>,
		pfeilradius, pfeilfarbe)
	#declare theta = theta + thetastep;
#end

#macro thetatangente(theta)
	arrow(kugel(theta, 0),
		kugel(theta, 0) - pfeillaenge * <cos(theta), -sin(theta), 0 >,
		pfeilradius, pfeilfarbe)
#end

#declare theta = 0;
#while (theta <= thetamax)
	thetatangente(theta)
	#declare theta = theta + thetastep;
#end

union {
	#declare theta = pi / 2;
	#declare phimin = 0;
	#declare phimax = pi / 2;
	#declare phistep = (phimax - phimin)/100;
	#declare phi = phimin;
	sphere { kugel(theta, phi), 0.7 * punktradius }
	#while (phi < phimax - phistep/2)
		cylinder {
			kugel(theta, phi),
			kugel(theta, phi + phistep),
			0.7 * punktradius
		}
		#declare phi = phi + phistep;
		sphere { kugel(theta, phi), 0.7 * punktradius }
	#end
	#declare thetamin = 0;
	#declare thetamax = pi / 2;
	#declare theta = thetamin;
	#declare thetastep = (thetamax - thetamin) / 100;
	#while (theta < thetamax - thetastep/2)
		cylinder {
			kugel(theta, phimin),
			kugel(theta + thetastep, phimin),
			0.7 * punktradius
		}
		cylinder {
			kugel(theta, phimax),
			kugel(theta + thetastep, phimax),
			0.7 * punktradius
		}
		#declare theta = theta + thetastep;
		sphere { kugel(theta, phimin), 0.7 * punktradius }
		sphere { kugel(theta, phimax), 0.7 * punktradius }
	#end
	
	sphere { e1, punktradius }
	sphere { e2, punktradius }
	sphere { e3, punktradius }
	pigment {
		color wegfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#declare gridfarbe = rgb<0.2,0.6,1.0>;
#declare gridradius = 0.005;

#macro breitenkreis(theta)
union {
	#declare phimin = 0;
	#declare phimax = 2 * pi;
	#declare phisteps = 200;
	#declare phistep = (phimax - phimin) / phisteps;
	#declare phi = phimin;
	#while (phi < phimax - phistep/2)
		cylinder {
			kugel(theta, phi),
			kugel(theta, phi + phistep),
			gridradius
		}
		sphere { kugel(theta, phi), gridradius }
		#declare phi = phi + phistep;
	#end
	pigment {
		color gridfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
#end

breitenkreis(pi/12)
breitenkreis(2 * pi/12)
breitenkreis(3 * pi/12)
breitenkreis(4 * pi/12)
breitenkreis(5 * pi/12)
breitenkreis(6 * pi/12)

#macro laengenkreis(phi)
union {
	#declare thetamin = 0;
	#declare thetamax = pi;
	#declare thetastep = (thetamax - thetamin) / 100;
	#declare theta = thetamin;
	sphere { kugel(theta, phi), gridradius }
	#while (theta < thetamax - thetastep/2)
		cylinder {
			kugel(theta, phi),
			kugel(theta + thetastep, phi),
			gridradius
		}
		#declare theta = theta + thetastep;
		sphere { kugel(theta, phi), gridradius }
	#end
	pigment {
		color gridfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
#end

#declare PHIstep = pi / 12;
#declare PHImin = 0;
#declare PHImax = 2 * pi;
#declare PHI = PHImin;
#while (PHI < PHImax - PHIstep/2)
	laengenkreis(PHI)
	#declare PHI = PHI + PHIstep;
#end

#declare winkelfarbe = rgb<0.6,0.2,1.0>;

intersection {
	cylinder { < 1.005, 0, 0>, <0.995, 0, 0>, 0.2 }
	plane { <0, -1, 0>, 0 }
	plane { <0, 0, -1>, 0 }
	pigment {
		color winkelfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
