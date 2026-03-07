import sys
import logging
import asyncio
from engineer_a.models import FieldBoundary
from engineer_a.api_soilgrids_wcs import _fetch_soilgrids_sync

logging.basicConfig(level=logging.INFO)

b = FieldBoundary.from_geojson({
    'type':'FeatureCollection',
    'features':[{
        'type':'Feature',
        'properties':{},
        'geometry':{'type':'Polygon','coordinates':[[[-93.593740, 41.516540],[-93.590000, 41.516540],[-93.590000, 41.513000],[-93.593740, 41.513000],[-93.593740, 41.516540]]]}
    }]
})

try:
    print("STARTING TEST")
    res = _fetch_soilgrids_sync(b)
    print("SUCCESS")
    print(res.head())
except Exception as e:
    import traceback
    traceback.print_exc()
