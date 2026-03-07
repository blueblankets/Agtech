from owslib.wcs import WebCoverageService
import logging
logging.basicConfig(level=logging.DEBUG)

wcs = WebCoverageService('https://maps.isric.org/mapserv?map=/map/bdod.map', version='2.0.1')
try:
    res = wcs.getCoverage(
        identifier='bdod_5-15cm_Q0.5', 
        subsets=[('X', -93.593740, -93.590000), ('Y', 41.513000, 41.516540)], 
        format='image/tiff', 
        crs='http://www.opengis.net/def/crs/EPSG/0/4326'
    )
    print("SUCCESS")
except Exception as e:
    with open("err.txt", "w") as f:
        f.write(str(e))
