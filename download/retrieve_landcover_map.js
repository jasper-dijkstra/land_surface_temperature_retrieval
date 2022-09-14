var tiles_asset = ee.FeatureCollection('users/yoannmalbeteau/jasper_roi_conus')
var geometry = tiles_asset.geometry()
print(geometry)

var dataset = ee.ImageCollection("ESA/WorldCover/v100").first();

var LC = dataset.clip(geometry)
print(LC)

var visualization = {
  bands: ['Map'],
};

Map.centerObject(LC);

Map.addLayer(LC, visualization, "Landcover");


// Export a cloud-optimized GeoTIFF.
Export.image.toDrive({
  image: LC,
  description: 'landcover_ESA',
  crsTransform: [0.009009009009009008931,0,-180,0,-0.009009009009009008931,60],
  crs: 'EPSG:4326',
  region: geometry,
  fileFormat: 'GeoTIFF',
  maxPixels: 1000000000,
  formatOptions: {
    cloudOptimized: true
  }
});
