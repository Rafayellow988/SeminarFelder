//
// gaussrand.pov -- Rand-Situation im Satz von Gauss
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<33, 20, 50>, <0, 0, 0>, 16/9, 0.02)
lightsource(<40, 50, 10>, 1, White)

#declare randgitterfarbe = rgb<1.0,0.8,0.4>;
#declare gitterfarbe = rgb<1.0,0.8,0.8>;
#declare zylinderfarbe = rgb<0.8,0.8,0.8>;
#declare zrad = 0.87;

#declare zoffset = 0.0;

arrow(-1.2 * e1 - zoffset * e2, 1.2  * e1 - zoffset * e2, 0.01, White)
arrow(-2.1 * e2,                0.65 * e2,                0.01, White)
arrow(-1.2 * e3 - zoffset * e2, 1.2  * e3 - zoffset * e2, 0.01, White)

#declare gridradius = 0.008;

#macro randgitter()
union {
	#declare Xstep = 0.2;
	#declare Xmin = -1;
	#declare Xmax = 1;
	#declare X = Xmin;
	#while (X < Xmax)
		cylinder { < Xmin, 0, X >, < Xmax, 0, X >, gridradius }
		cylinder { < X, 0, Xmin >, < X, 0, Xmax >, gridradius }
		#declare X = X + Xstep;
	#end
}
#end

intersection {
	cylinder { < 0, -2, 0 >, < 0, 1, 0>, zrad }
	randgitter()
	pigment {
		color randgitterfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

union {
	#declare Zmin = -0.2;
	#declare Zmax = -2;
	#declare Zstep = 0.2;
	#declare Z = Zmin;
	#declare Xstep = 0.2;
	#declare Xmin = -0.8;
	#declare Xmax = 0.8;
	#declare Ystep = 0.2;
	#while (Z > Zmax)
		#declare X = Xmin;
		#while (X < Xmax)
			#declare Ymin = -floor(sqrt(zrad*zrad-X*X)/Ystep)*Ystep;
			#declare Ymax = -Ymin;
			cylinder {
				< X, Z, Ymin >,
				< X, Z, Ymax >,
				gridradius
			}
			cylinder {
				< Ymin, Z, X >,
				< Ymax, Z, X >,
				gridradius
			}
			#declare X = X + Xstep;
		#end
		#declare Z = Z - Zstep;
	#end
	#declare Ymin = -1;
	#declare Ymax = 1;
	#declare Ystep = 0.2;
	#declare Xmin = -1;
	#declare Xmax = 1;
	#declare Xstep = 0.2;
	#declare Y = Ymin;
	#while (Y < Ymax - Ystep/2)
		#declare X = Xmin;
		#while (X < Xmax - Xstep/2)
			#if (sqrt(X*X+Y*Y) < zrad)
				cylinder {
					< X, 0, Y >,
					< X, -2, Y >,
					gridradius
				}
			#end
			#declare X = X + Xstep;
		#end
		#declare Y = Y + Ystep;
	#end
	pigment {
		color gitterfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

cylinder { < 0, 0, 0>, < 0, 0.001, 0 >, zrad
	pigment {
		color rgbt<0.4,0.8,0.6,0.2>
	}
	no_shadow
}

cylinder { < 0, 0, 0>, < 0, -2, 0>, zrad
	pigment {
		color rgbt<0.4,0.8,0.6,0.8>
	}
}
