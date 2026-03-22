/***************************************
 * Ecotone Degradation Zone Delineation
 * Purpose: Delineation of a critical ecological zone based on
 * vegetation vulnerability (V) and water stress (W)
 *
 * Baseline period: 2000–2010 (11 years)
 * Platform: Google Earth Engine (GEE) Code Editor
 * Access: https://code.earthengine.google.com/
 *
 * Data sources:
 *   - MODIS MOD13A2 (NDVI)
 *   - CHIRPS Daily (precipitation)
 *   - TerraClimate (PET)
 ***************************************/
// ==============================
// STEP 1 (No admin boundary): Define a ROI around Yuyang/Yulin using coordinates
// ==============================

// A rough ROI around Yuyang District (Yulin, Shaanxi)
// You can adjust later after seeing it on the map.
var roi = ee.Geometry.Rectangle([108.8, 38.6, 110.0, 39.6]);
// Format: [minLon, minLat, maxLon, maxLat]

// Show ROI on map
Map.centerObject(roi, 8);
Map.addLayer(roi, {color: 'red'}, 'ROI (Yuyang/Yulin approx)');

// Print ROI info
print('ROI:', roi);
print('ROI area (km2):', roi.area().divide(1e6));
// ==============================
// STEP 1.2: Quick check with land cover (MCD12Q1)
// ==============================

// Use MODIS Land Cover Type 1 (IGBP) as a quick reference
var lc2005 = ee.ImageCollection("MODIS/061/MCD12Q1")
  .filterDate('2005-01-01', '2005-12-31')
  .first()
  .select('LC_Type1')
  .clip(roi);

// Display land cover
Map.addLayer(lc2005, {}, 'Land Cover (LC_Type1, 2005)');

// Highlight "barren or sparsely vegetated" class (IGBP = 16)
var barren = lc2005.eq(16).selfMask();
Map.addLayer(barren, {palette: ['yellow']}, 'Barren/Sparse (IGBP=16)');


// ==============================
// STEP 2: NDVI Mean (2000–2010)
// ==============================

// Load MOD13A2 NDVI
var ndviCol = ee.ImageCollection("MODIS/061/MOD13A2")
  .filterDate('2000-01-01', '2010-12-31')
  .select('NDVI')
  .map(function(img){
    return img.multiply(0.0001)
              .copyProperties(img, ['system:time_start']);
  });

// Compute mean NDVI
var ndviMean = ndviCol.mean().clip(roi);

// Display
Map.addLayer(ndviMean, 
  {min: 0, max: 0.6, palette: ['brown','yellow','green']},
  'NDVI Mean (2000–2010)'
);

// Print mean value
var meanNDVI = ndviMean.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 1000,
  maxPixels: 1e13
});

print('Mean NDVI in ROI:', meanNDVI);



// ==============================
// STEP 3.1 (Fix): NDVI StdDev & CV with explicit band names
// ==============================

// Make sure mean NDVI has a clear band name
var ndviMean1 = ndviCol.mean().rename('ndvi_mean').clip(roi);

// Compute stdDev with explicit band name
var ndviStd1 = ndviCol.reduce(ee.Reducer.stdDev())
  .rename('ndvi_std')
  .clip(roi);

// CV = std / mean
var ndviCV1 = ndviStd1.divide(ndviMean1).rename('ndvi_cv').clip(roi);

// Visualize
Map.addLayer(ndviMean1, {min: 0, max: 0.6, palette: ['brown','yellow','green']}, 'NDVI Mean (2000–2010) [fixed]');
Map.addLayer(ndviStd1,  {min: 0, max: 0.2, palette: ['white','orange','red']}, 'NDVI StdDev (2000–2010) [fixed]');
Map.addLayer(ndviCV1,   {min: 0, max: 1.0, palette: ['white','orange','red']}, 'NDVI CV (2000–2010) [fixed]');

// Print summary stats
print('Mean NDVI (ROI) [fixed]:', ndviMean1.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 1000,
  maxPixels: 1e13
}));

print('Mean NDVI StdDev (ROI) [fixed]:', ndviStd1.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 1000,
  maxPixels: 1e13
}));

print('Mean NDVI CV (ROI) [fixed]:', ndviCV1.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 1000,
  maxPixels: 1e13
}));



// ==============================
// STEP 4 (Robust): Percentiles + V mask
// ==============================

// Percentiles for ndvi_mean and ndvi_cv (computed within ROI)
var percentiles = ee.Image.cat([ndviMean1, ndviCV1])
  .reduceRegion({
    reducer: ee.Reducer.percentile([30, 70]),
    geometry: roi,
    scale: 1000,
    maxPixels: 1e13
  });

print('Percentiles dict:', percentiles);
print('Percentiles keys:', percentiles.keys());

// Extract thresholds (use the keys printed above if needed)
var p30_mean = ee.Number(percentiles.get('ndvi_mean_p30'));
var p70_cv   = ee.Number(percentiles.get('ndvi_cv_p70'));

print('p30_mean:', p30_mean);
print('p70_cv:', p70_cv);

// Build V mask
var Vmask = ndviMean1.lte(p30_mean)
  .and(ndviCV1.gte(p70_cv))
  .selfMask();

Map.addLayer(Vmask, {palette: ['cyan']}, 'V mask (low mean & high CV)');



// ==============================
// STEP 5 (Most Stable): Water stress W using CHIRPS P + TerraClimate PET
// AI = P_annual / PET_annual  (both in mm/year)
// ==============================

// 1) Precipitation: CHIRPS daily (mm/day) -> annual total (mm/yr) -> mean over 2000–2010
var P_daily = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2000-01-01', '2010-12-31')
  .select('precipitation');

var years = ee.List.sequence(2000, 2010);

var P_annual = ee.ImageCollection.fromImages(
  years.map(function(y){
    y = ee.Number(y);
    var start = ee.Date.fromYMD(y, 1, 1);
    var end   = start.advance(1, 'year');
    var total = P_daily.filterDate(start, end).sum().rename('P_annual'); // mm/yr
    return total.set('year', y);
  })
).mean().clip(roi);  // mm/yr

// 2) PET: TerraClimate monthly PET (mm/month) -> annual total (mm/yr) -> mean over 2000–2010
var PET_monthly = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
  .filterDate('2000-01-01', '2010-12-31')
  .select('pet'); // mm/month

var PET_annual = ee.ImageCollection.fromImages(
  years.map(function(y){
    y = ee.Number(y);
    var start = ee.Date.fromYMD(y, 1, 1);
    var end   = start.advance(1, 'year');
    var total = PET_monthly.filterDate(start, end).sum().rename('PET_annual'); // mm/yr
    return total.set('year', y);
  })
).mean().clip(roi);

// 3) AI = P / PET
var AI = P_annual.divide(PET_annual).rename('AI').clip(roi);

// Visualize AI (red=drier, green=wetter)
Map.addLayer(AI, {min: 0, max: 1.2, palette: ['red','yellow','green']}, 'AI (CHIRPS/TerraClimate) mean 2000-2010');

// 4) P30 threshold for AI within ROI
var aiP = AI.reduceRegion({
  reducer: ee.Reducer.percentile([30]),
  geometry: roi,
  scale: 4000,      // TerraClimate 分辨率较粗，用 4000 更匹配且更快
  maxPixels: 1e13
});

print('AI percentiles:', aiP);
print('AI percentile keys:', aiP.keys());

// Usually key is "AI" in your setup
var ai_p30 = ee.Number(aiP.get('AI'));
print('AI_p30:', ai_p30);

// 5) W mask: low AI = higher water stress
var Wmask = AI.lte(ai_p30).selfMask();
Map.addLayer(Wmask, {palette: ['blue']}, 'W mask (AI <= P30)');



// ==============================
// STEP 6: Combine V and W
// ==============================

// Intersection (both vulnerable vegetation AND water stress)
var Dmask = Vmask.and(Wmask).selfMask();

Map.addLayer(Dmask, {palette: ['purple']}, 'Degradation core (V AND W)');



// ==============================
// STEP 7: Convert degradation core to vector polygon
// ==============================

var studyArea = Dmask
  .reduceToVectors({
    geometry: roi,
    scale: 1000,
    geometryType: 'polygon',
    eightConnected: true,
    labelProperty: 'zone',
    maxPixels: 1e13
  });

Map.addLayer(studyArea, {color: 'black'}, 'Final Study Area');

// Calculate total area (km2)
// Calculate total area (km2) by summing feature areas (robust)
var studyAreaWithArea = studyArea.map(function(f){
  var a = f.geometry().area(100).divide(1e6); // errorMargin=100 meters
  return f.set('area_km2', a);
});

print('Polygon count:', studyAreaWithArea.size());
print('Area summary (km2):', studyAreaWithArea.aggregate_sum('area_km2'));



// ==============================
// STEP 8: Keep largest connected patch
// ==============================

// Add area property again (safety)
var withArea = studyArea.map(function(f){
  return f.set('area_km2', f.geometry().area(100).divide(1e6));
});

// Sort descending by area
var sorted = withArea.sort('area_km2', false);

// Get largest polygon
var largestPatch = ee.Feature(sorted.first());

Map.addLayer(largestPatch, {color: 'yellow'}, 'Largest Patch');

// Print area
print('Largest patch area (km2):', largestPatch.get('area_km2'));


// ==============================
// STEP 9: Merge ALL degradation patches into one modeling boundary
// ==============================

// 1) Dissolve all patches into a single geometry (most robust)
var mergedGeom = studyArea.geometry().dissolve(100);  // errorMargin=100m

// 2) OPTIONAL: Smooth boundary (recommended for SD modeling boundary)
// buffer(+) then buffer(-) removes small holes and jagged edges
var mergedSmooth = mergedGeom.buffer(200).buffer(-200);

// 3) Display
Map.addLayer(mergedSmooth, {color: 'lime'}, 'Merged Study Area (smoothed)');

// 4) Area (km2)
var mergedArea_km2 = mergedSmooth.area(100).divide(1e6);
print('Merged Study Area area (km2):', mergedArea_km2);



// ==============================
// EXPORT FINAL STUDY AREA
// ==============================

// 1) Convert geometry to FeatureCollection
var finalStudyArea = ee.FeatureCollection([
  ee.Feature(mergedSmooth).set({
    'NDVI_mean_p30': p30_mean,
    'NDVI_CV_p70': p70_cv,
    'AI_p30': ai_p30,
    'Area_km2': mergedArea_km2
  })
]);

Map.addLayer(finalStudyArea, {color: 'white'}, 'Final Study Area (export)');

Export.table.toDrive({
  collection: finalStudyArea,
  description: 'Yulin_Degradation_StudyArea_GeoJSON',
  fileFormat: 'GeoJSON'
});









// ======================================
// EXPORT FIGURE DATA FOR PUBLICATION
// ======================================

// 1️⃣ NDVI Mean
Export.image.toDrive({
  image: ndviMean,
  description: 'Figure 2 (a) NDVI_mean_2000_2010',
  folder: 'GEE_StudyArea',
  fileNamePrefix: 'NDVI_mean_2000_2010',
  scale: 1000,
  region: roi,
  maxPixels: 1e13
});

// 2️⃣ NDVI CV
Export.image.toDrive({
  image: ndviCV1,
  description: 'Figure 2 (b) NDVI_CV_2000_2010',
  folder: 'GEE_StudyArea',
  fileNamePrefix: 'NDVI_CV_2000_2010',
  scale: 1000,
  region: roi,
  maxPixels: 1e13
});

// 3️⃣ Aridity Index (AI)
Export.image.toDrive({
  image: AI,
  description: 'Figure 3 AI_2000_2010',
  folder: 'GEE_StudyArea',
  fileNamePrefix: 'AI_2000_2010',
  scale: 4000,
  region: roi,
  maxPixels: 1e13
});

// 4️⃣ Degradation Core Mask
Export.image.toDrive({
  image: Dmask,
  description: 'Figgue 4 Degradation_Core',
  folder: 'GEE_StudyArea',
  fileNamePrefix: 'Degradation_Core',
  scale: 1000,
  region: roi,
  maxPixels: 1e13
});

var bounds = mergedSmooth.bounds();
print('Bounding Box:', bounds);
