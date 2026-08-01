def find_geotif_file_prefix_match(target_geotiff, candidate_files):
    for b in candidate_files:
        if target_geotiff.replace(".json", "").lower() == b.replace(".json", "").lower():
            return b
    return None
