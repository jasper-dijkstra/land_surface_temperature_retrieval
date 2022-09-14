var tiles_asset = ee.FeatureCollection('users/yoannmalbeteau/jasper_roi_conus')
var geometry = tiles_asset.geometry()
print(geometry)
var dataset = ee.Image('CGIAR/SRTM90_V4');
var elevation = dataset.select('elevation');
var slope = ee.Terrain.slope(elevation);

var ele = elevation.clip(geometry)
print(ele)
Map.setCenter(-112.8598, 36.2841, 8);
Map.addLayer(ele, {min: 0, max: 2000}, 'elevation');
Map.addLayer(slope, {min: 0, max: 5}, 'slope');

// Export a cloud-optimized GeoTIFF.
Export.image.toDrive({
  image: slope,
  description: 'DEM_SRTM90_V4b',
  crsTransform: [0.009009009009009008931,0,-180,0,-0.009009009009009008931,60],
  crs: 'EPSG:4326',
  region: geometry,
  fileFormat: 'GeoTIFF',
  maxPixels: 1000000000,
  formatOptions: {
    cloudOptimized: true
  }
});
