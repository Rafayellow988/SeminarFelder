let r = 0;

let thetaMaxSlider, phiMaxSlider;
let densitySlider;
let thetaMax, phiMax;
let density = 50;

function setup(){
  createCanvas(700, 700, WEBGL);//size(600, 400);
  angleMode(DEGREES);
  colorMode(HSB);
  stroke(321, 38, 80);
  strokeWeight(2);
  noFill();

  r = width/4;

}

function draw(){
  background(230, 50, 15);
  orbitControl(4, 4);//Mouse control

  rotateY(90);
  rotateZ(65);
  for(let phi = 0; phi < 360; phi += 360/density){
    beginShape();
    for(let theta = 0; theta < 360; theta += 360/density){
      let x = r * cos(phi);
      let y = r * sin(phi) * sin(theta);
      let z = r * sin(phi) * cos(theta);
      point(x, y, z);
    }
    endShape(CLOSE);
  }

  let mappedDensity = int(map(density, 13, 72, 1, 60));
}