import os
import json
import argparse
import rasterio
import numpy as np
import yaml

from alive_progress import alive_bar
from modeling.DataMap import ColorMap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with Model.")
    parser.add_argument(
        "--imagery_path_map", type=str, help="Path to the orthomosaic path map."
    )
    parser.add_argument(
        "--output_ortho_folder",
        type=str,
        help="Path to the folder which will store the output orthomosaics.",
    )
    parser.add_argument(
        "--preds_folder",
        type=str,
        help="Path to the folder that contains the predictions.",
    )
    parser.add_argument(
        "--hyperparameters_file",
        type=str,
        help="Path to the file that contains the hyperparameters file.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=4,
        help="The number of channels in the orthomosaic.",
    )
    parser.add_argument(
        "--write_imagery",
        action="store_true",
        help="Flag to indicate if imagery should be generated.",
    )
    args = parser.parse_args()
    with open(args.imagery_path_map, "r") as f:
        imagery_path_map = json.load(f)

    print("Reading hyperparameters...")
    hyperparameters = {}
    with open(args.hyperparameters_file) as stream:
        try:
            hyperparameters = yaml.safe_load(stream)
            print(hyperparameters)
        except yaml.YAMLError as exc:
            print("Encountered error parsing hyperparameters...")
            print(exc)
    output_label_2_rgba_color_map = ColorMap(
        hyperparameters["channel_maps"]["model_class_2_color_map"],
        hyperparameters["channel_maps"]["output_class_2_idx_map"],
    )

    for orthomosaic_path in imagery_path_map.values():
        ortho_name = os.path.split(orthomosaic_path)[-1].split(".")[0]
        print("Loading external base geotif from:", orthomosaic_path)
        input_geotiff_data = rasterio.open(orthomosaic_path, "r")
        channel_data = []
        for i in range(0, args.channels):
            if args.write_imagery:
                channel_data.append(input_geotiff_data.read(i + 1))
            else:
                channel_data.append(
                    np.zeros(
                        [input_geotiff_data.height, input_geotiff_data.width],
                        dtype=np.uint8,
                    )
                )
        print("Done...")

        red_lookup = np.vectorize(
            lambda x: output_label_2_rgba_color_map.getColorDict(x)["red"]
        )
        green_lookup = np.vectorize(
            lambda x: output_label_2_rgba_color_map.getColorDict(x)["green"]
        )
        blue_lookup = np.vectorize(
            lambda x: output_label_2_rgba_color_map.getColorDict(x)["blue"]
        )
        alpha_lookup = np.vectorize(
            lambda x: output_label_2_rgba_color_map.getColorDict(x)["alpha"]
        )

        print("Loading NPZ data...")
        files = os.listdir(args.preds_folder)
        with alive_bar(len(files)) as bar:
            for file in files:
                if file.endswith(".npz"):
                    data = np.load(os.path.join(args.preds_folder, file))
                    if data["orthomosaic_name"] == ortho_name:
                        width, height = data["preds"].shape
                        x = data["x"]
                        y = data["y"]

                        actual_width = max(
                            0, min(width, -1 * (y - input_geotiff_data.height))
                        )
                        actual_height = max(
                            0, min(height, -1 * (x - input_geotiff_data.width))
                        )

                        if args.write_imagery:
                            alpha_mask = (
                                alpha_lookup(data["preds"])[
                                    :actual_width, :actual_height
                                ]
                                == 255
                            )
                            channel_data[0][y : y + height, x : x + width][
                                alpha_mask
                            ] = (
                                channel_data[0][y : y + height, x : x + width][
                                    alpha_mask
                                ]
                                * 0.6
                                + red_lookup(data["preds"])[
                                    :actual_width, :actual_height
                                ][alpha_mask]
                                * 0.4
                            )
                            channel_data[1][y : y + height, x : x + width][
                                alpha_mask
                            ] = (
                                channel_data[1][y : y + height, x : x + width][
                                    alpha_mask
                                ]
                                * 0.6
                                + green_lookup(data["preds"])[
                                    :actual_width, :actual_height
                                ][alpha_mask]
                                * 0.4
                            )
                            channel_data[2][y : y + height, x : x + width][
                                alpha_mask
                            ] = (
                                channel_data[2][y : y + height, x : x + width][
                                    alpha_mask
                                ]
                                * 0.6
                                + blue_lookup(data["preds"])[
                                    :actual_width, :actual_height
                                ][alpha_mask]
                                * 0.4
                            )
                        else:
                            channel_data[0][y : y + height, x : x + width] = red_lookup(
                                data["preds"]
                            )[:actual_width, :actual_height]
                            channel_data[1][y : y + height, x : x + width] = (
                                green_lookup(data["preds"])[
                                    :actual_width, :actual_height
                                ]
                            )
                            channel_data[2][y : y + height, x : x + width] = (
                                blue_lookup(data["preds"])[
                                    :actual_width, :actual_height
                                ]
                            )
                            channel_data[3][y : y + height, x : x + width] = (
                                alpha_lookup(data["preds"])[
                                    :actual_width, :actual_height
                                ]
                            )
                    data = None

                imagery_str = "imagery" if args.write_imagery else "no-imagery"
                output_orthomosaic_path = os.path.join(
                    args.output_ortho_folder,
                    "review_" + imagery_str + "_" + os.path.split(orthomosaic_path)[-1],
                )
            # pylint: disable-next=not-callable
            bar()
        print("Done...")
        print("Writing output orthomosaic...")
        with rasterio.open(
            output_orthomosaic_path,
            "w",
            driver="GTiff",
            height=input_geotiff_data.height,
            width=input_geotiff_data.width,
            count=args.channels,
            dtype="uint8",
            crs=input_geotiff_data.crs,
            compress=None if args.write_imagery else "lzw",
            transform=input_geotiff_data.transform,
        ) as dst:
            for i, band in enumerate(channel_data):
                dst.write(band, i + 1)
        print("Done")
