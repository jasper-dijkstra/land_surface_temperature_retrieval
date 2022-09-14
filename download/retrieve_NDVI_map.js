var tiles_asset = ee.FeatureCollection('users/yoannmalbeteau/jasper_roi_conus')
var geometry = tiles_asset.geometry()
print(geometry)

var dataset = ee.ImageCollection('MODIS/061/MOD13A2')
                  .filter(ee.Filter.date('2020-06-01', '2022-06-01'));
var ndvi = dataset.select('NDVI');

var mean = ndvi.reduce(ee.Reducer.mean());
mean = mean.clip(geometry)

var std = ndvi.reduce(ee.Reducer.stdDev());
std = std.clip(geometry)


var ndviVis = {
  min: 0.0,
  max: 9000.0,
  palette: [
    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
    '012E01', '011D01', '011301'
  ],
};
Map.addLayer(mean, ndviVis, 'NDVI');
Map.addLayer(std, ndviVis, 'std');

// Export a cloud-optimized GeoTIFF.
Export.image.toDrive({
  image: mean,
  description: 'NDVI_MOD13A2_mean',
  crsTransform: [0.009009009009009008931,0,-180,0,-0.009009009009009008931,60],
  crs: 'EPSG:4326',
  region: geometry,
  fileFormat: 'GeoTIFF',
  maxPixels: 1000000000,
  formatOptions: {
    cloudOptimized: true
  }
});
