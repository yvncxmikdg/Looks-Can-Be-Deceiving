import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from modeling.utils.sample_generator_utils import draw_objects_on_mask


def inspect_image(color_data):
    result = []
    for sample_index in range(0, color_data.shape[0]):
        result.append(color_data[sample_index])
    return result


def get_polygon_outline(polygon, outline_width):
    return polygon.buffer(outline_width).difference(polygon)


def inspect_polygons_on_image(
    batched_color_data,
    batched_unadjusted_buildings,
    batched_adjusted_buildings,
    batched_gt_buildings,
    downsample_rate=1.0,
    outline_width=2.0,
):
    elems = []
    for batch_idx, color_data in enumerate(batched_color_data):
        unadjusted_buildings_mask = draw_objects_on_mask(
            batched_unadjusted_buildings[batch_idx],
            0,
            0,
            color_data.shape[0],
            color_data.shape[1],
            int(color_data.shape[0] * (1.0 / downsample_rate)),
            int(color_data.shape[1] * (1.0 / downsample_rate)),
            channels=3,
            color_accessor=lambda x: tuple([255, 0, 0]),
            geometry_accessor=lambda x: get_polygon_outline(
                x.getGeometry("pixels"), outline_width
            ),
        )
        adjusted_buildings_mask = draw_objects_on_mask(
            batched_adjusted_buildings[batch_idx],
            0,
            0,
            color_data.shape[0],
            color_data.shape[1],
            int(color_data.shape[0] * (1.0 / downsample_rate)),
            int(color_data.shape[1] * (1.0 / downsample_rate)),
            channels=3,
            color_accessor=lambda x: tuple([0, 0, 255]),
            geometry_accessor=lambda x: get_polygon_outline(
                x.getGeometry("pixels"), outline_width
            ),
        )
        gt_buildings_mask = draw_objects_on_mask(
            batched_gt_buildings[batch_idx],
            0,
            0,
            color_data.shape[0],
            color_data.shape[1],
            int(color_data.shape[0] * (1.0 / downsample_rate)),
            int(color_data.shape[1] * (1.0 / downsample_rate)),
            channels=3,
            color_accessor=lambda x: tuple([0, 255, 0]),
            geometry_accessor=lambda x: get_polygon_outline(
                x.getGeometry("pixels"), outline_width
            ),
        )
        tmp = torch.tensor(
            gt_buildings_mask + unadjusted_buildings_mask + adjusted_buildings_mask
        ).to(color_data.get_device())
        outline_mask = torch.sum(tmp, dim=-1) == 0.0
        masked_color_data = color_data * torch.stack(
            [outline_mask, outline_mask, outline_mask], dim=-1
        ).to(color_data.get_device())

        elem_result = masked_color_data + tmp
        elems.append(torch.tensor(elem_result))

    return torch.stack(elems, dim=0).cpu().numpy()


def inspect_mask(query_data, axis=2, downsample_rate=1.0):
    _, x_dim, y_dim = query_data.shape
    downsampled_query_data = (
        F.interpolate(
            query_data.unsqueeze(1),
            size=[int(x_dim * downsample_rate), int(y_dim * downsample_rate)],
            mode="nearest",
        )
        .squeeze()
        .cpu()
        .detach()
        .numpy()
    )
    result = []
    for sample_index in range(0, downsampled_query_data.shape[0]):
        sample_query_data = downsampled_query_data[sample_index]
        sample_query_data = (sample_query_data * 255).astype(np.uint8)
        query = np.stack(
            [sample_query_data, sample_query_data, sample_query_data], axis=axis
        )
        result.append(np.uint8(query))
    return result


def inspect_labels(label_data, cmap, axis=2, downsample_rate=1.0):
    _, x_dim, y_dim = label_data.shape
    downsampled_label_data = (
        F.interpolate(
            label_data.unsqueeze(1).float(),
            size=[int(x_dim * downsample_rate), int(y_dim * downsample_rate)],
            mode="nearest",
        )
        .squeeze()
        .cpu()
        .detach()
    )

    if len(downsampled_label_data.shape) == 2:
        downsampled_label_data = downsampled_label_data.unsqueeze(0)

    downsampled_label_data = downsampled_label_data.numpy()

    result = []
    for sample_index in range(0, downsampled_label_data.shape[0]):
        sample_label_data = downsampled_label_data[sample_index]
        color_map_r = np.vectorize(lambda x: cmap.getColorDict(idx=x)["red"])
        color_map_g = np.vectorize(lambda x: cmap.getColorDict(idx=x)["green"])
        color_map_b = np.vectorize(lambda x: cmap.getColorDict(idx=x)["blue"])

        label_image_r = color_map_r(sample_label_data)
        label_image_g = color_map_g(sample_label_data)
        label_image_b = color_map_b(sample_label_data)
        label = np.stack([label_image_r, label_image_g, label_image_b], axis=axis)

        result.append(np.uint8(label))
    return result


def inspect_grad_flow(named_parameters, title_suffix=None, nan_eps=1.0e-10):
    fig, _ = plt.subplots(figsize=(15, 10))
    layers = []
    avg_grads = []
    avg_weights = []

    nan_check_x = []
    nan_check_y = []

    nan_check_grad_x = []
    nan_check_grad_y = []

    max_grads = []
    min_grads = []
    max_weights = []
    min_weights = []

    for num_layer, (n, p) in enumerate(named_parameters):
        if p.requires_grad and "bias" not in n and not p.grad is None:
            avg_grad = p.grad.abs().mean().detach().cpu()
            avg_weight = p.abs().mean().detach().cpu()
            layers.append(n)

            if torch.isnan(avg_weight):
                print("\tFOUND NaN WEIGHT AT LAYER: ", n)
                nan_check_x.append(num_layer)
                nan_check_y.append(nan_eps)  # temp value to plotting
                avg_weight = nan_eps

            avg_weights.append(avg_weight)
            max_weights.append(p.abs().max().detach().cpu())
            min_weights.append(p.abs().min().detach().cpu())

            if torch.isnan(avg_grad):
                print("\tFOUND NaN GRADIENT AT LAYER: ", n)
                nan_check_grad_x.append(num_layer)
                nan_check_grad_y.append(nan_eps)
                avg_grad = nan_eps

            avg_grads.append(avg_grad)
            max_grads.append(p.grad.abs().max().detach().cpu())
            min_grads.append(p.grad.abs().min().detach().cpu())

    plt.plot(avg_weights, alpha=0.3, color="r", label="Mean(Abs(Weights))")
    plt.plot(max_weights, alpha=0.3, color="m", label="Max(Abs(Weights))")
    plt.plot(min_weights, alpha=0.3, color="pink", label="Min(Abs(Weights))")

    plt.plot(avg_grads, alpha=0.3, color="b", label="Mean(Abs(Gradients))")
    plt.plot(max_grads, alpha=0.3, color="c", label="Max(Abs(Gradients))")
    plt.plot(min_grads, alpha=0.3, color="orange", label="Min(Abs(Gradients))")

    plt.scatter(nan_check_x, nan_check_y, color="r", marker="X")
    plt.scatter(nan_check_grad_x, nan_check_grad_y, color="r", marker="X")
    plt.yscale("log")
    plt.hlines(0, 0, len(avg_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(avg_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient & Weight" + title_suffix if title_suffix else "")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    try:
        plt.close('all')
    except RuntimeError:
        pass
    return data
