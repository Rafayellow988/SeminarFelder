//
// 3dimage.pov -- -template for 3d images rendered by Povray
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

#declare kugelfarbe = rgb<0.6,0.8,1.0>;
#declare kurvenfarbe = rgb<0.8,0.4,0.6>;
#declare netzfarbe = rgb<1.0,0.8,0.4>;

place_camera(<33, 20, 50>, <0, 0, 0>, 16/9, 0.037)
lightsource(<40, 15, -10>, 1, 0.7 * White)
lightsource(<4, -1, 20>, 1, 0.3 * White)

global_settings { ambient_light rgb<2, 2, 2> }

arrow(-1.2 * e1, 1.2 * e1, 0.01, White)
arrow(-1.2 * e2, 1.2 * e2, 0.01, White)
arrow(-1.2 * e3, 1.2 * e3, 0.01, White)

sphere { <0, 0, 0>, 1
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.50
	}
}

#declare k = 0.12;
#macro loxodrome(l)
	(1 / cosh(k * l)) * < cos(l), sinh(k * l), sin(l) >
#end

#declare kurvendurchmesser = 0.012;

union {
	#declare phistep = pi / 72;
	#declare phi = 0;
	#while (phi < 10 * pi)
		sphere { loxodrome(-phi), kurvendurchmesser }
		cylinder {
			loxodrome(-phi),
			loxodrome(-phi - phistep),
			kurvendurchmesser
		}
		sphere { loxodrome(phi), kurvendurchmesser }
		cylinder {
			loxodrome(phi),
			loxodrome(phi + phistep),
			kurvendurchmesser
		}
		#declare phi = phi + phistep;
	#end
	pigment {
		color kurvenfarbe
	}
	finish {
		metallic
		specular 0.99
	}
	rotate < 0, 0, 0>
}

#macro kugel(phi, theta)
	< sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi) >
#end

#declare netzradius = 0.004;

union {
	#declare phi = 0;
	#declare phistep = pi / 12;
	#declare thetastep = pi / 144;
	#while (phi < 2 * pi - phistep/2)
		#declare theta = 0;
		sphere { kugel(phi, theta), netzradius }
		#while (theta < pi - thetastep/2)
			cylinder {
				kugel(phi, theta),
				kugel(phi, theta + thetastep),
				netzradius
			}
			#declare theta = theta + thetastep;
			sphere { kugel(phi, theta), netzradius }
		#end
		#declare phi = phi + phistep;
	#end
	#declare thetastep = pi / 12;
	#declare phistep = pi / 144;
	#declare theta = thetastep;
	#while (theta < pi - thetastep / 2)
		#declare phi = 0;
		#while (phi < 2 * pi - phistep/2)
			sphere { kugel(phi, theta), netzradius }
			cylinder {
				kugel(phi, theta),
				kugel(phi + phistep, theta),
				netzradius
			}
			#declare phi = phi + phistep;
		#end
		#declare theta = theta + thetastep;
	#end
	pigment {
		color netzfarbe
	}
	finish {
		metallic
		specular 0.99
	}
}
