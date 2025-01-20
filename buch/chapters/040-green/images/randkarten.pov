//
// randkarten.pov -- -template for 3d images rendered by Povray
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<33, 20, 50>, <0, 0.34, 0>, 16/9, 0.022)
lightsource(<10, 50, 40>, 1, White)

#declare rotgitterfarbe = rgb<0.8,0.4,0.6>;
#declare blaugitterfarbe = rgb<0.4,0.6,0.8>;
#declare mischfarbe = rgb<0.6,0.4,0.8>;
#declare kugelfarbe = rgb<0.8,0.8,0.8>;

//arrow(-e1, e1, 0.01, White)
//arrow(-e2, e2, 0.01, White)
//arrow(-e3, e3, 0.01, White)

intersection {
	sphere { <0,0,0>, 1 }
	plane { <0,-1,0>, 0 }
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#declare phi1 = function(U,V) { U/4 + pi/6 - V*V/30 }
#declare theta1 = function(U, V) { V/4 + V*(U+2)*(U+2)/100 }

#declare phi2 = function(U, V) { U/4 + 2.2*pi/6 + 0.05 }
#declare theta2 = function(U, V) { (1-U/20-U*U/30) * V/3.7 }

#macro kugelpunkt1(U, V)
	<	cos(theta1(U,V)) * cos(phi1(U,V)),
		sin(theta1(U, V)),
		cos(theta1(U,V)) * sin(phi1(U, V)) >
#end

#macro kugelpunkt2(U, V)
	<	cos(theta2(U,V)) * cos(phi2(U,V)),
		sin(theta2(U, V)),
		cos(theta2(U,V)) * sin(phi2(U, V)) >
#end

#declare gitterradius = 0.006;
#declare gitterfaktor = 1.7;

union {
	#declare U = -2;
	#while (U < 2.1)
		#declare V = 0;
		#declare Vstep = 0.1;
		sphere { kugelpunkt1(U, V), gitterradius }
		#while (V < 2.5 - Vstep/2)
			cylinder {
				kugelpunkt1(U, V),
				kugelpunkt1(U, V + Vstep),
				gitterradius
			}
			#declare V = V + Vstep;
			sphere { kugelpunkt1(U, V), gitterradius }
		#end
		#declare U = U + 0.5;
	#end
	#declare V = 0;
	#declare gr = gitterfaktor * gitterradius;
	#declare Ustep = 0.01;
	#while (V < 2.6)
		#declare U = -2;
		#declare Ustep = 0.1;
		sphere { kugelpunkt1(U, V), gr }
		#while (U < 2 - Ustep/2)
			cylinder {
				kugelpunkt1(U, V),
				kugelpunkt1(U + Ustep, V),
				gr
			}
			#declare U = U + Ustep;
			sphere { kugelpunkt1(U, V), gr }
		#end
		#declare V = V + 0.5;
		#declare gr = gitterradius;
		#declare Ustep = 0.1;
	#end
	pigment {
		color rotgitterfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

union {
	#declare U = -2;
	#while (U < 2.1)
		#declare V = 0;
		#declare Vstep = 0.1;
		sphere { kugelpunkt2(U, V), gitterradius }
		#while (V < 2.5 - Vstep/2)
			cylinder {
				kugelpunkt2(U, V),
				kugelpunkt2(U, V + Vstep),
				gitterradius
			}
			#declare V = V + Vstep;
			sphere { kugelpunkt2(U, V), gitterradius }
		#end
		#declare U = U + 0.5;
	#end
	#declare V = 0;
	#declare gr = gitterfaktor * gitterradius;
	#declare Ustep = 0.01;
	#while (V < 2.6)
		#declare U = -2;
		sphere { kugelpunkt2(U, V), gr }
		#while (U < 2 - Ustep/2)
			cylinder {
				kugelpunkt2(U, V),
				kugelpunkt2(U + Ustep, V),
				gr
			}
			#declare U = U + Ustep;
			sphere { kugelpunkt2(U, V), gr }
		#end
		#declare V = V + 0.5;
		#declare gr = gitterradius;
		#declare Ustep = 0.1;
	#end
	pigment {
		color blaugitterfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#macro randpunkt(U) 
	< cos(U), 0, sin(U) >
#end

union {
	#declare gr = 1.1 * gitterfaktor * gitterradius;
	#declare Umin = phi2(-2, 0);
	#declare Umax = phi1(2, 0);
	#declare Ustep = (Umax - Umin) / 100;
	#declare U = Umin;
	sphere { randpunkt(U), gr }
	#while (U < Umax - Ustep/2)
		cylinder {
			randpunkt(U),
			randpunkt(U + Ustep),
			gr
		}
		#declare U = U + Ustep;
		sphere { randpunkt(U), gr }
	#end
	pigment {
		color mischfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

union {
	#declare gr = 0.99 * gitterfaktor * gitterradius;
	#declare Umin = 0;
	#declare Umax = 2 * pi;
	#declare Ustep = (Umax - Umin) / 200;
	#declare U = Umin;
	sphere { randpunkt(U), gr }
	#while (U < Umax - Ustep/2)
		cylinder {
			randpunkt(U),
			randpunkt(U + Ustep),
			gr
		}
		#declare U = U + Ustep;
		sphere { randpunkt(U), gr }
	#end
	pigment {
		color kugelfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
