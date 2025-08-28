//
// 3dimage.pov -- -template for 3d images rendered by Povray
//
// (c) 2023 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

place_camera(<43, 20, -50>, <0, 0, 0>, 16/9, 0.04)
lightsource(<-10, 30, -40>, 10, 0.5 * White)
lightsource(<30, 30, -40>, 10, 0.5 * White)

#declare sx = 1.4;
#declare sy = 1.0;
#declare sz = 0.7;

#declare w = 20;

#declare l = 0.5;

#declare kurvenradius = 0.02;
#declare kurvenfarbe = rgb<0.8,0,0>;
#declare tangentialvektorfarbe = rgb<0.8,0.2,0.8>;

arrow(-2*e1, 2*e1, 0.015, White)
arrow(-(sz+0.3)*e2, (sz+0.3)*e2, 0.015, White)
arrow(-2*e3, 2*e3, 0.015, White)

sphere { O, 1
	pigment {
		color rgb<0.8,0.8,0.8>
	}
	finish {
		metallic
		specular 0.9
	}
	scale < sx, sz, sy >
}

intersection {
	box { <-2,-sz - 0.2, -2>, <2, sz + 0.2, 2> }
	plane { < (1/sx) * cos(radians(w)), 0, (1/sy) * sin(radians(w)) >, 0.001 }
	plane { -< (1/sx) * cos(radians(w)), 0, (1/sy) * sin(radians(w)) >, 0.001 }
	pigment {
		color rgbt<0.6,0.8,1.0,0.7>
	}
	finish {
		metallic
		specular 0.9
	}
	no_shadow
}

intersection {
	box { <-l,0, -l>, <l, 2, l> }
	plane { < 0, 1, 0>, sz+0.001 }
	plane { -< 0, 1, 0>, -sz+0.001 }
	pigment {
		color rgbt<0.6,0.8,1.0,0.7>
	}
	finish {
		metallic
		specular 0.9
	}
	no_shadow
	
}

#declare meridianbreite = 0.01;

cylinder { < meridianbreite/2, 0, 0>, <-meridianbreite/2, 0, 0> , 1
	pigment {
		color rgb<0.6,0.4,0.2>
	}
	finish {
		metallic
		specular 0.9
	}
	scale < 1, 1.001 * sz, 1.001 * sy >
}
cylinder { < 0, meridianbreite/2, 0>, < 0, -meridianbreite/2, 0> , 1
	pigment {
		color rgb<0.6,0.4,0.2>
	}
	finish {
		metallic
		specular 0.9
	}
	scale < 1.001 * sx, 1.0 * sz, 1.001 * sy >
}
cylinder { < 0, 0, meridianbreite/2>, < 0, 0, -meridianbreite/2> , 1
	pigment {
		color rgb<0.6,0.4,0.2>
	}
	finish {
		metallic
		specular 0.9
	}
	scale < 1.001 * sx, 1.001 * sz, 1 * sy >
}

#macro schnittkurvenpunkt(phi)
	< sx * sin(phi) * sin(radians(w)), sz * cos(phi), -sy * sin(phi) * cos(radians(w)) >
#end


union {
	#declare phistart = 0;
	#declare phiend = pi;
	#declare phisteps = 50;
	#declare phistep = pi/phisteps;
	#declare phi = phistart;
	#declare p = schnittkurvenpunkt(phi);
	sphere { p, kurvenradius }
	#while (phi < phiend - phistep/2)
		#declare pold = p;
		#declare phi = phi + phistep;
		#declare p = schnittkurvenpunkt(phi);
		sphere { p, kurvenradius }
		sphere { -p, kurvenradius }
		cylinder { pold, p, kurvenradius }
		cylinder { -pold, -p, kurvenradius }
	#end
	pigment {
		color rgb<0.8,0,0>
	}
	finish {
		metallic
		specular 0.9
	}
}

arrow(<0, sz, 0>, <0, sz, 0> +<sx * sin(radians(w)), 0, -cos(radians(w)) >,
	kurvenradius, tangentialvektorfarbe)
