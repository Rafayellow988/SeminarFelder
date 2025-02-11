//
// greenrand.inc
//
// (c) 2025 Prof Dr Andreas Müller
//
#include "../../../common/common.inc"
#include "colors.inc"
#include "greenranddata.inc"

place_camera(<15, 10, -33>, <0, 0.35, 0>, 16/9, 0.057)
lightsource(<-10, 35, -10>, 1, White)

#declare flaechenfarbe = rgb<0.4,0.6,0.8>;
#declare kurvefarbe = rgb<0.0,0.6,0.0>;
#declare gitterfarbe = rgb<1.0,0.8,0.2>;
#declare wandfarbe = rgb<0.2,0.8,0.6>;

arrow(-1.6 * e1, 1.6 * e1, 0.01, White)
arrow(-0.1 * e2, 0.8 * e2, 0.01, White)
arrow(-0.1 * e3, 1.3 * e3, 0.01, White)

