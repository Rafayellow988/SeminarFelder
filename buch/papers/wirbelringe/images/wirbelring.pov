//
// wirbelring.pov
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"

#declare wirbellinie = rgb<0.8,0.8,0.8>;
#declare wirbelradius = 0.01;

place_camera(<33, 20, 50>, <0, 0, 0>, 16/9, 0.023)
lightsource(<40, 50, 10>, 1, White)

#macro kreispunkt(phi)
	< cos(phi), 0, sin(phi) >
#end

union {
#declare phistart = radians(0);
#declare phiend = radians(360);
#declare phisteps = 300;
#declare phistep = (phiend - phistart) / phisteps;
#declare phi = phistart;
#declare p = kreispunkt(phi);
	sphere { p, wirbelradius }
#while (phi <= phiend)
	#declare pold = p;
	#declare phi = phi + phistep;
	#declare p = kreispunkt(phi);
	cylinder { pold, p, wirbelradius }
	sphere { p, wirbelradius }
#end
	//cone { kreispunkt(0), 2 * wirbelradius, kreispunkt(0) + <0, 0, 0.1>, 0}
	pigment {
		color wirbellinie
	}
	finish {
		metallic
			 0.99
	}
}

// Tangent vector

#declare tangent_phi = radians(20); // Change this value for different positions
#declare tangent_length = 0.3;      // Length of the tangent arrow
#declare tangent_color = rgb<1,0,1>; // Arrow color

#declare tangent_point = kreispunkt(tangent_phi);
#declare tangent_dir = < -sin(tangent_phi), 0, cos(tangent_phi) >;

arrow(
    tangent_point,
    tangent_point + tangent_length * tangent_dir,
    0.01, // thickness
    tangent_color
)

#declare wirbelthickness = 0.007;
#declare rohrradius = 0.27;

#macro wirbel(phi, farbe)
union {
	#declare thetastart = radians(0);
	#declare thetaend = radians(360);
	#declare thetasteps = 24;
	#declare thetastep = (thetaend - thetastart) / thetasteps;
	#declare theta = thetastart;
	#while (theta < thetaend - thetastep/2)
		#declare richtung = 0.24 * rohrradius * < sin(theta), -cos(theta), 0 >;
		#declare p = < 1 + rohrradius * cos(theta), rohrradius * sin(theta), 0>;
		arrow(p, p + richtung, wirbelthickness, farbe)
		#declare theta = theta + thetastep;
	#end
	pigment {
		color farbe
	}
	finish {
		metallic
		specular 0.99
	}
	rotate <0, degrees(phi), 0>
}
#end

#declare wfarbe = rgb<0.8,0,0>;

#declare phistart = radians(50);
#declare phiend = radians(360);
#declare phisteps = 72;
#declare phistep = (phiend - phistart) / phisteps;
#declare phi = phistart;
#while (phi < phiend - phistep/2)
	#declare wfarbe = CHSV2RGB(<degrees(phi), 1, 1, 0, 0>);
	wirbel(-phi + radians(1), wfarbe)
	#declare phi = phi + phistep;
#end


