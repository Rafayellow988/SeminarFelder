//
// parallel.pov -- Paralleltransport für verschiedene Karten
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<33, 7, 10>, <0, -0.116, 0>, 16/9, 0.0456)
lightsource(<40, 25, 10>, 1, White)

//arrow(-1.2 * e1, 1.2 * e1, 0.01, White)
//arrow(-1.2 * e2, 1.2 * e2, 0.01, White)
//arrow(-1.2 * e3, 1.2 * e3, 0.01, White)

#declare O = <0, 0, 0>;
#declare kugelfarbe = rgb<0.8,0.8,0.8>;
#declare kartesischfarbe = rgb<0.2,0.6,1.0>;
#declare polarfarbe = rgb<1.0,0.4,1.0>;

intersection {
	difference {
		sphere { O, 1 }
		sphere { O, 0.999 }
	}
	cylinder { <0, 0, 0>, <2, 0, 0>, 0.8 }
	//box { <0,-0.7,-0.7>, <1.7,0.7,0.7> }
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#macro kartesisch(X, Y)
	< sqrt(1-X*X-Y*Y), Y, X >
#end

#declare gridradius = 0.005;

union {
	#declare Xmin = -0.5;
	#declare Xstep = 0.1;
	#declare Xmax = 0.5;
	#declare X = Xmin;
	#declare Ymin = -0.5;
	#declare Ymax = 0.5;
	#declare Ysteps = 100;
	#declare Ystep = (Ymax - Ymin) / Ysteps;
	#while (X < Xmax + Xstep/2)
		#declare Y = Ymin;
		sphere { kartesisch(X, Y), gridradius }
		#while (Y < Ymax - Ystep/2)
			cylinder {
				kartesisch(X, Y),
				kartesisch(X, Y + Ystep),
				gridradius

			}
			#declare Y = Y + Ystep;
			sphere { kartesisch(X, Y), gridradius }
		#end
		#declare X = X + Xstep;
	#end
	#declare Xsteps = 100;
	#declare Xstep = (Xmax - Xmin) / Xsteps;
	#declare Ystep = 0.1;
	#declare Y = Ymin;
	#while (Y < Ymax + Ystep/2)
		#declare X = Xmin;
		sphere { kartesisch(X, Y), gridradius }
		#while (X < Xmax - Xstep/2)
			cylinder {
				kartesisch(X, Y),
				kartesisch(X + Xstep, Y),
				gridradius

			}
			#declare X = X + Xstep;
			sphere { kartesisch(X, Y), gridradius }
		#end
		#declare Y = Y + Ystep;
	#end
	
	pigment {
		color kartesischfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#declare awinkel = -1.05;

#macro polar(phi, r)
	<
		cos(r) * cos(awinkel) - sin(r) * sin(awinkel) * cos(phi),
		sin(r) * sin(phi),
		sin(r) * cos(awinkel) * cos(phi) + cos(r) * sin(awinkel)
	>
#end

union {
	#declare Rmin = 0.5;
	#declare Rmax = 1.5;
	#declare Rstep = 0.1;
	#declare Phimin = -0.5;
	#declare Phimax = 0.5;
	#declare Phisteps = 100;
	#declare Phistep = (Phimax - Phimin) / Phisteps;
	#declare R = Rmin;
	#while (R < Rmax + Rstep/2)
		#declare Phi = Phimin;
		sphere { polar(Phi, R), gridradius }
		#while (Phi < Phimax - Phistep/2)
			cylinder {
				polar(Phi, R),
				polar(Phi + Phistep, R),
				gridradius
			}
			#declare Phi = Phi + Phistep;
			sphere { polar(Phi, R), gridradius }
		#end
		#declare R = R + Rstep;
	#end
	#declare Phistep = 0.1;
	#declare Rsteps = 100;
	#declare Rstep = (Rmax - Rmin) / Rsteps;
	#declare Phi = Phimin;
	#while (Phi < Phimax + Phistep/2)
		#declare R = Rmin;
		sphere { polar(Phi, R), gridradius }
		#while (R < Rmax - Rstep/2)
			cylinder {
				polar(Phi, R),
				polar(Phi, R + Rstep),
				gridradius
			}
			#declare R = R + Rstep;
			sphere { polar(Phi, R), gridradius }
		#end
		#declare Phi = Phi + Phistep;
	#end
	pigment {
		color polarfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
