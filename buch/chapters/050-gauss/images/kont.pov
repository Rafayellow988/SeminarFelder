//
// 3dimage.pov -- -template for 3d images rendered by Povray
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<2, 5, -15>, <0, 0, 0>, 16/9, 0.092)
lightsource(<-10, 50, -40>, 1, White)

arrow(<-0.6,-0.5,-0.5>, <0.85, -0.5, -0.5>, 0.01, White)
arrow(<-0.5,-0.5,-0.6>, <-0.5, -0.5, 0.85>, 0.01, White)
arrow(<-0.5,-0.6,-0.5>, <-0.5, 0.85, -0.5>, 0.01, White)

//arrow(-0.80 * e2, 0.80 * e2, 0.01, White)
//arrow(-0.85 * e3, 0.85 * e3, 0.01, White)

#declare blockfarbe = rgbt<0.8,0.8,0.8,0.2>;
#declare vektorfarbe = rgb<0.2,0.6,1.0>;
#declare pfarbe = rgb<0.8,0.4,0.8>;

#macro vektor(X, Y, Z)
	0.3 * <X + 0.3, Y*Y + 0, Z + X*X + 0>
#end

#macro feld(X, Y, Z)
	arrow(<X, Y, Z>, <X, Y, Z> + vektor(X, Y, Z), 0.01, vektorfarbe)
#end

#macro pfeld(X, Y, Z, e)
	#declare V = vektor(X, Y, Z);
	arrow(<X, Y, Z>, <X, Y, Z> + e * vdot(e, V), 0.01, pfarbe)
#end

#declare balkenradius = 0.002;

#macro balken(X, Y, Z, e)
	#declare V = vektor(X, Y, Z);
	#declare V1 = <X, Y, Z> + V;
	#declare V2 = <X, Y, Z> + e * vdot(e, V);
	union {
		cylinder { V1, V2, balkenradius }
		sphere { V1, balkenradius }
		sphere { V2, balkenradius }
		pigment {
			color White
		}
		finish {
			metallic
			specular 0.9
		}
	}
#end

box { < -0.5, -0.5, -0.5 >, < 0.5, 0.5, 0.5 >
	pigment {
		color blockfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}

#declare Astep = 0.2;
#declare Bstep = 0.2;
#declare Amin = -0.4;
#declare Amax = 0.4;
#declare Bmin = -0.4;
#declare Bmax = 0.4;

union {
	#declare A = Amin;
	#while (A < Amax + Astep/2)
		#declare B = Bmin;
		#while (B < Bmax + Bstep/2)
			feld(   A,    B,  0.5)
			feld(   A,    B, -0.5)
			feld(   A,  0.5,    B)
			feld(   A, -0.5,    B)
			feld( 0.5,    A,    B)
			feld(-0.5,    A,    B)
			pfeld(   A,    B,  0.5, <0, 0, 1>)
			pfeld(   A,    B, -0.5, <0, 0, 1>)
			pfeld(   A,  0.5,    B, <0, 1, 0>)
			pfeld(   A, -0.5,    B, <0, 1, 0>)
			pfeld( 0.5,    A,    B, <1, 0, 0>)
			pfeld(-0.5,    A,    B, <1, 0, 0>)
			//balken(   A,    B,  0.5, <0, 0, 1>)
			//balken(   A,    B, -0.5, <0, 0, 1>)
			//balken(   A,  0.5,    B, <0, 1, 0>)
			//balken(   A, -0.5,    B, <0, 1, 0>)
			//balken( 0.5,    A,    B, <1, 0, 0>)
			//balken(-0.5,    A,    B, <1, 0, 0>)
			#declare B = B + Bstep;
		#end
		#declare A = A + Astep;
	#end
}

union {
	cylinder { < 0.495, -0.5, -0.5 >, < 0.505, -0.5, -0.5 >, 0.02 }
	cylinder { < -0.5, 0.495, -0.5 >, < -0.5, 0.505, -0.5 >, 0.02 }
	pigment {
		color White
	}
	finish {
		metallic
		specular 0.9
	}
}
