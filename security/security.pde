import processing.video.*;

Capture cam;
PImage dream;
File dreamFile;

void setup() {
  size(1024, 768);

  String[] cameras = Capture.list();
  
  if (cameras.length == 0) {
    println("There are no cameras available for capture.");
    exit();
  } else {
    println("Available cameras:");
    for (int i = 0; i < cameras.length; i++) {
      println(cameras[i]);
    }
    
    // The camera can be initialized directly using an 
    // element from the array returned by list():
    cam = new Capture(this, cameras[0]);
    cam.start();     
  }      
}

void draw() {
  if (cam.available() == true) {
    cam.read();
  }
  image(cam, 0, 0);
  // The following does the same, and is faster when just drawing the image
  // without any additional resizing, transformations, or tint.
  // set(0, 0, cam);
  
  if (frameCount % 24 == 0) {
    save("data/frames/screenshot.jpg");
  }
  
  dreamFile = new File(dataPath("frames/screenshot-dream.jpg"));
  if (dreamFile.exists() && frameCount % 10 == 0) {
    dream = loadImage("data/frames/screenshot-dream.jpg");
    delay(20);
    image(dream, 0, 0);
  }
}