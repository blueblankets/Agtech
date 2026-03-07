import traceback

try:
    from mapie.regression import MapieRegressor
    print("Mapie imported successfully!")
except Exception as e:
    with open('err_utf8.log', 'w', encoding='utf-8') as f:
        f.write(f"Exception: {e}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
