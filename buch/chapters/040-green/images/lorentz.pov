//
// lorentz.pov -- Lorentz-Kraft
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<30, 20, 50>, <0, 0, 0>, 16/9, 0.02)
lightsource(<10, 5, 40>, 1, White)

#declare Bfeldfarbe = rgb<0.8,0.2,0.2>;
#declare Vfarbe = rgb<0.2,0.6,1.0>;
#declare Ffarbe = rgb<0,0.6,0>;

#declare feldradius = 0.007;

#declare h = 0.2;
#declare l = 1.5;
#declare X = -l * h;
#while (X < l * h + h/2)
	#declare Y = -l * h;
	#while (Y < l * h + h/2)
		#declare P = < X, 0, Y>;
		arrow(P + <0,-0.5,0>, P + <0,0.5,0>, feldradius, Bfeldfarbe)
		#declare Y = Y + h;
	#end
	#declare X = X + h;
#end

//arrow(-e1, e1, 0.01, White)
//arrow(-e2, e2, 0.01, White)
//arrow(-e3, e3, 0.01, White)

#macro bahn(T)
	< T*T, -0.2*T*T*T-0.3*T, T>
#end

#declare bahnradius = 0.005;

union {
	#declare Y = -1;
	#declare Ystep = 0.01;
	#while (Y < 1 + Ystep/2)
		cylinder { bahn(Y), bahn(Y+Ystep), bahnradius }
		sphere { bahn(Y), bahnradius }
		#declare Y = Y + Ystep;
	#end
	sphere { bahn(Y), bahnradius }
	pigment {
		color Vfarbe
	}
	finish {
		metallic
		specular 0.9
	}
}
#declare B = <0, 1, 0>;
#declare V = <0,-0.3,1>;
arrow(<0,0,0>, 0.7 * V, 0.01, Vfarbe)

arrow(<0,0,0>, -vcross(0.7*V, B), 0.01, Ffarbe)

sphere { <0, 0, 0>, 0.02
	pigment {
		color rgb<1,0.8,0.2>
	}
	finish {
		metallic
		specular 0.9
	}
}
