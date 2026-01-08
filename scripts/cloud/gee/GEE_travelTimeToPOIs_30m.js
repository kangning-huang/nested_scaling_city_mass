/**** 0. CONFIG ****/
var POIs = ee.FeatureCollection('projects/ee-knhuang/assets/Paris_pois_9cats');
// var region = POIs.geometry().bounds().buffer(50000);

// compute bbox of all POIs
var region = POIs.geometry().bounds();

var crs = 'EPSG:32631';  // UTM zone 31N for Paris
// var crs = 'EPSG:4326';  // UTM zone 31N for Paris
var scale = 30;

Map.centerObject(region, 12);
Map.addLayer(POIs, {color: 'black'}, 'POIs (all)');

/**** 1. LAND COVER ****/
var lc = ee.Image('ESA/WorldCover/v100/2020').select('Map');

var lcClasses =   [10,   20,   30,   40,   50,   60,   70,   80,   90,   95,   100];
var speedValues = [3.06, 3.60, 4.86, 2.50, 5.00, 3.00, 2.00, 0.01, 4.86, 4.20, 2.00];

var baseSpeed = lc.remap(lcClasses, speedValues, 3.0);  // default 3 km/h
var landMask = lc.neq(80);
baseSpeed = baseSpeed.updateMask(landMask);

/**** 2. TOPOGRAPHIC ADJUSTMENTS - FIXED ****/
var dsm = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2')
  .select('DSM')
  .filterBounds(region)
  .mosaic()
  .setDefaultProjection(crs, null, scale)  // KEY FIX: set projection
  .clip(region);

Map.addLayer(dsm, {min: 0, max: 500}, 'DEBUG: DSM', false);

// Now slope will work
var slopeDeg = ee.Terrain.slope(dsm);
Map.addLayer(slopeDeg, {min: 0, max: 30}, 'DEBUG: Slope', false);

// Elevation factor
var fElev = dsm.expression('1.016 * exp(-0.0001072 * elev)', {elev: dsm});

// Slope factor (Tobler)
var slopeRad = slopeDeg.multiply(Math.PI / 180);
var s = slopeRad.tan();
var hikingSpeed = s.expression('6 * exp(-3.5 * abs(s + 0.05))', {s: s});
var fSlope = hikingSpeed.divide(5);

// If slope is still problematic, provide fallback
fSlope = fSlope.unmask(1);  // default factor of 1 (no adjustment)
fElev = fElev.unmask(1);

var landSpeed = baseSpeed.multiply(fElev).multiply(fSlope);
landSpeed = landSpeed.where(landSpeed.lte(0), 0.5);
landSpeed = landSpeed.unmask(3);  // fallback 3 km/h

Map.addLayer(landSpeed.clip(region), {min: 0, max: 6, palette: ['red', 'yellow', 'green']}, 'DEBUG: Land Speed', false);

/**** 3. GRIP4 ROADS ****/
var grip4_europe = ee.FeatureCollection('projects/sat-io/open-datasets/GRIP4/Europe')
  .filterBounds(region);

var roadBuffer = grip4_europe.map(function(f) { return f.buffer(15); });

Map.addLayer(grip4_europe, {}, 'DEBUG: GRIP roads', false);
Map.addLayer(roadBuffer, {}, 'DEBUG: GRIP roads buffer', false);

var roadSpeedMotor = ee.Image(0)
  .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 1)), 105)
  .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 2)), 80)
  .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 3)), 60)
  .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 4)), 50)
  .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 5)), 40);

Map.addLayer(roadSpeedMotor, {}, 'DEBUG: road speed motor', false);

var roadSpeedWalk = ee.Image(0).paint(roadBuffer, 5);

Map.addLayer(roadSpeedWalk, {}, 'DEBUG: road speed walk', false);

/**** 4. FRICTION ****/
var walkSpeed = landSpeed.max(roadSpeedWalk).max(0.5);
var motorSpeed = landSpeed.max(roadSpeedMotor).max(0.5);

// var pixelKm = scale / 1000;  // 0.03 km per pixel
// var frictionWalk = ee.Image(60 * pixelKm).divide(walkSpeed).rename('friction_walk');
// var frictionMotor = ee.Image(60 * pixelKm).divide(motorSpeed).rename('friction_motor');

var frictionWalk = ee.Image(60 / 1000).divide(walkSpeed).rename('friction_walk');
var frictionMotor = ee.Image(60 / 1000).divide(motorSpeed).rename('friction_motor');


// Debug: check friction values are reasonable
// At 5 km/h: friction = 60 * 0.03 / 5 = 0.36 min/pixel
// At 100 km/h: friction = 60 * 0.03 / 100 = 0.018 min/pixel
Map.addLayer(frictionWalk.clip(region), {min: 0, max: 1, palette: ['blue', 'yellow', 'red']}, 'Friction Walk (min/pixel)', false);
Map.addLayer(frictionMotor.clip(region), {min: 0, max: 0.1, palette: ['blue', 'yellow', 'red']}, 'Friction Motor (min/pixel)', false);

/**** 5. RASTERIZE POIs ****/
function rasterizeCategory(cat) {
  var fc = POIs.filter(ee.Filter.eq('category', cat));
  var buffered = fc.map(function(f) { return f.buffer(50); });
  
  var src = ee.Image(0).byte()
    .paint(fc, 1)
    .selfMask();
  
  return src;
}

/**** 6. TRAVEL TIME ****/
var maxMinutes = 180;
var maxMinutes = 1000;

function travelTimeForCategory(cat) {
  var src = rasterizeCategory(cat);
  Map.addLayer(src.clip(region), {min: 0, max: 1, palette: ['white', 'red']}, 'DEBUG: Source ' + cat, false);
  
  var ttWalk = frictionWalk.cumulativeCost({
    source: src,
    maxDistance: maxMinutes
  }).rename('tt_walk_' + cat);
  
  var ttMotor = frictionMotor.cumulativeCost({
    source: src,
    maxDistance: maxMinutes
  }).rename('tt_motor_' + cat);
  
  return { walk: ttWalk, motor: ttMotor };
}

/**** 7. RUN ****/
var cat = 'eating';
var tt = travelTimeForCategory(cat);

Map.addLayer(tt.walk, {min: 0, max: 60, palette: ['green', 'yellow', 'red']}, 'Travel time WALK (min)');
Map.addLayer(tt.motor, {min: 0, max: 30, palette: ['blue', 'cyan', 'yellow', 'red']}, 'Travel time MOTOR (min)');