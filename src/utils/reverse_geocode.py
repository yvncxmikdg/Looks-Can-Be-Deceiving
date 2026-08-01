import requests

def get_address(longitude: float, latitude: float, apikey: str) -> str:
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    payload = {
        "latlng": f"{latitude},{longitude}",
        "key": apikey,
        "result_type": "street_address",
    }
    try:
        result = requests.get(endpoint, payload).json()
    except:
        return "REVERSE GEOCODING ERROR (POSSIBLY NETWORK CONNECTIVITY)"
    if not ("status" in result.keys()):
        return "REVERSE GEOCODING ERROR (INVALID API RESULT)"
    elif result["status"] == "OK":
        return result["results"][0]["formatted_address"]
    elif result["status"] == "ZERO_RESULTS":
        return "NO MATCHING ADDRESS"
    elif result["status"] == "REQUEST_DENIED":
        return "REVERSE GEOCODING ERROR (INVALID API KEY)"
    else:
        return "REVERSE GEOCODING ERROR (UNKNOWN CAUSE)"