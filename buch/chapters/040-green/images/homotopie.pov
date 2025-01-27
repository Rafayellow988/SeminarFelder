//
// 3dimage.pov -- -template for 3d images rendered by Povray
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

global_settings { ambient_light rgb<2, 2, 2> }

#declare O = <0, 0, 0>;
#declare w1 = radians(30);
#declare w2 = radians(60);

place_camera(<33, 20, 50>, <0, 0.324, 0>, 16/9, 0.0178)
lightsource(<5, 25, -10>, 1, 0.7 * White)

//arrow(-1.1 * e1, 1.1 * e1, 0.01, White)
//arrow(-1.1 * e2, 1.1 * e2, 0.01, White)
//arrow(-1.1 * e3, 1.1 * e3, 0.01, White)

#declare kugelfarbe = rgb<0.8,0.8,0.8>;

#macro kugel(theta, phi) 
	< cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi) >
#end

#declare phi = function(T, S) { radians(110) * T }
#declare theta = function(T, S) { 0.3 + 0.5 * T + 0.5 * S * sin(T * pi) - 0.4 * (1 - S) * (1 - S) * sin(2 * T * pi) * sin(2 * T * pi) * T }

difference {
	sphere { O, 1 }
	union {
		sphere { O, 0.99 }
		cylinder { < 0, -1.1, 0>, <0, 1.1, 0>, 0.4 }
		plane { <0, 1, 0>, 0.1 }
		plane { < cos(w1), 0, sin(w1) >, 0 }
		plane { < cos(w2), 0, sin(w2) >, 0 }
	}
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.95
	}
}

#macro kurve(S, farbe, kurvenradius)
union {
	#declare Tmin = 0;
	#declare Tmax = 1;
	#declare Tstep = (Tmax - Tmin) / 100;
	#declare T = Tmin;
	sphere { kugel(theta(T, S), phi(T, S)), kurvenradius }
	#while (T < Tmax - Tstep/2)
		cylinder {
			kugel(theta(T, S), phi(T, S)),
			kugel(theta(T+Tstep, S), phi(T+Tstep, S)),
			kurvenradius
		}
		#declare T = T + Tstep;
		sphere { kugel(theta(T, S), phi(T, S)), kurvenradius }
	#end
	pigment {
		color farbe
	}
	finish {
		metallic
		specular 0.99
	}
}
#end

#declare darkred = rgb<0.8,0,0>;
#declare blau = rgb<0,0,1>;
#declare violet = rgb<0.8,0.2,0.8>;

#declare kr = 0.008;

kurve(0, darkred, kr)
kurve(2.8/4, violet, kr)
kurve(1, blau, kr)

#macro Skurve(T, farbe, kurvenradius)
union {
	#declare Smin = 0;
	#declare Smax = 1;
	#declare Sstep = (Smax - Smin) / 100;
	#declare S = Smin;
	sphere { kugel(theta(T, S), phi(T, S)), kurvenradius }
	#while (S < Smax - Sstep/2)
		cylinder {
			kugel(theta(T, S), phi(T, S)),
			kugel(theta(T, S + Sstep), phi(T, S + Sstep)),
			kurvenradius
		}
		#declare S = S + Sstep;
		sphere { kugel(theta(T, S), phi(T, S)), kurvenradius }
	#end
	pigment {
		color farbe
	}
	finish {
		metallic
		specular 0.99
	}
}
#end

#declare gelb = rgb<1,0.8,0>;

#declare SS = 0.1;
#declare SSstep = 0.1;
#while (SS < 1 - SSstep/2)
	Skurve(SS, gelb, 0.7 * kr)
	#declare SS = SS + SSstep;
#end

union {
	sphere { kugel(theta(0, 0), phi(0, 0)), 2 * kr }
	sphere { kugel(theta(1, 0), phi(1, 0)), 2 * kr }
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.99
	}
}
