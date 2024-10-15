import ee
import geemap

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize()

# Define region of interest 
roi = ee.Geometry.Polygon(
    [[[-122.5, 37.0], [-122.5, 37.5], [-121.5, 37.5], [-121.5, 37.0]]]
)

# Define parameters
START_DATE = '2022-01-01'
END_DATE = '2022-12-31'
NDVI_BANDS = ['B8', 'B4']  # NIR and Red for NDVI

# Fetch Sentinel-2 images, filtered by date and region
sentinel = ee.ImageCollection('COPERNICUS/S2') \
    .filterBounds(roi) \
    .filterDate(START_DATE, END_DATE) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

# Add NDVI to each image
def add_ndvi(image):
    ndvi = image.normalizedDifference(NDVI_BANDS).rename('NDVI')
    return image.addBands(ndvi)

ndvi_collection = sentinel.map(add_ndvi).select('NDVI')

# Harmonic regression setup
timeField = 'system:time_start'

def add_harmonic_terms(image):
    # Compute time in fractional years
    date = ee.Date(image.get(timeField))
    years = date.difference(ee.Date('1970-01-01'), 'year')
    
    # Add time-based terms for harmonic regression
    timeRadians = years.multiply(2 * ee.Number(math.pi))
    image = image.addBands(timeRadians.cos().rename('cos'))
    image = image.addBands(timeRadians.sin().rename('sin'))
    return image.addBands(ee.Image.constant(1).rename('constant')).addBands(years.rename('t'))

# Apply harmonic terms and linear regression
harmonic_landsat = ndvi_collection.map(add_harmonic_terms)

# Reduce the collection by fitting a harmonic model
harmonic_reducer = ee.Reducer.linearRegression(4, 1)  # 4 independent variables: constant, t, cos, sin
harmonic_trend = harmonic_landsat.reduce(harmonic_reducer)

# Extract the coefficients
coefficients = harmonic_trend.select('coefficients').arrayProject([0]).arrayFlatten([['constant', 't', 'cos', 'sin']])

# Compute phase, amplitude, and mean NDVI
phase = coefficients.select('cos').atan2(coefficients.select('sin'))
amplitude = coefficients.select('cos').hypot(coefficients.select('sin'))
mean_ndvi = harmonic_landsat.select('NDVI').reduce(ee.Reducer.mean())

# Map results to HSV color space
hsv_image = phase.unitScale(-math.pi, math.pi).addBands(amplitude.unitScale(0, 1)).addBands(mean_ndvi.unitScale(0, 1)).hsvToRgb()

# Display results
Map = geemap.Map()
Map.centerObject(roi, 10)
Map.addLayer(hsv_image, {}, 'HSV Image: Phase, Amplitude, Mean NDVI')
Map
