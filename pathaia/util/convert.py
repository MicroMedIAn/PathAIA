"""
A Module to translate pathaia csv to json annotation for micromap.

It uses networkx to handle graph structures.
"""
import json
import pandas as pd
import networkx as nx
import openslide
import os
from .types import PathLike
from typing import Dict, Iterable, Tuple
import warnings


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class BottomLeftNotFound(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class NextPointNotFound(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class NoPathFound(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class OutOfBound(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


colorCycle = [
    "#f44336", "#8bc34a", "#ffeb3b", "#673ab7", "#e91e63", "#cddc39", "#9c27b0",
    "#ffc107", "#3f51b5", "#ff9800", "#2196f3", "#ff5722", "#03a9f4", "#795548",
    "#00bcd4", "#607d8b", "#009688", "#4caf50"
]


def handle_predicted_patches(
    patch_file: PathLike,
    level: int,
    column: str
):
    """
    Read a patch file.

    Read lines of the patch csv looking for 'column' label.
    Args:
        patch_file (str): absolute path to a csv patch file.
        level (int): pyramid level to query patches in the csv.
        column (str): header of the column to use to label individual patches.
    Yields:
        tuple: position and label of patches (x, y, label).

    """
    df = pd.read_csv(patch_file)
    level_df = df[df["level"] == level]
    for _, row in level_df.iterrows():
        yield row["x"], row["y"], row["dx"], row[column], column


def get_category(
    val: float,
    thresholds: Dict[int, Tuple[float, float]]
):
    """
    Return the break-apart categorical label from estimation.

    *********************************************************
    """
    for label, bounds in thresholds.items():
        low, high = bounds
        if val >= low and val < high:
            return label
    raise OutOfBound(
        "Value: {} is out of bounds for thresholds: {}!".format(val, thresholds)
    )


def gen_categorical_from_floatpreds(
    patch_file: PathLike,
    level: int,
    column: str,
    thresholds: Dict[int, Tuple[float, float]]
):
    """
    Yield categorical patches from float predictions.

    *************************************************
    """
    for patch in handle_predicted_patches(
        patch_file, level, column
    ):
        x, y, d, val, author = patch
        try:
            yield x, y, d, get_category(val, thresholds), author
        except OutOfBound:
            pass


def get_categorical_layer_edges(
    categorical_patch_generator: Iterable,
    color_dict: Dict[int, str],
    classname_dict: Dict[int, str]
):
    """
    Create graph features from patches for every layer of annotation.

    *******************************************************************
    """
    layer_nodes = dict()
    layer_edges = dict()
    layer_meta = dict()
    interval = None
    for patch in categorical_patch_generator:
        x, y, d, cl, author = patch
        if interval is None:
            interval = d
        cl_name = classname_dict[cl]
        if cl_name not in layer_nodes:
            layer_nodes[cl_name] = set()
            layer_meta[cl_name] = {
                "label": "{}_{}".format(author, cl_name),
                "author": "{}".format(author),
                "text": "",
                "color": color_dict[cl],
                "date": ""
            }
        # (x, y) is just the top left corner, to plot the polygon,
        # we will need the four corners
        layer_nodes[cl_name].add((x, y))
        layer_nodes[cl_name].add((x + interval, y))
        layer_nodes[cl_name].add((x, y + interval))
        layer_nodes[cl_name].add((x + interval, y + interval))
    for layer, nodes in layer_nodes.items():
        layer_edges[layer] = set()
        for node in nodes:
            x, y = node
            for neighbor in [
                (x + interval, y),
                (x - interval, y),
                (x, y + interval),
                (x, y - interval),
                (x - interval, y - interval),
                (x - interval, y + interval),
                (x + interval, y + interval),
                (x + interval, y - interval)
            ]:
                if neighbor in nodes:
                    layer_edges[layer].add((node, neighbor))
    return layer_edges, layer_meta, interval


def get_categorical_segments_from_edges(layer_edges: Dict):
    """
    Create segments from layer edges.

    *********************************
    """
    layer_segments = dict()
    for layer, edges in layer_edges.items():
        layer_segments[layer] = []
        # create a graph
        layer_graph = nx.Graph()
        layer_graph.add_edges_from(edges)
        for c in nx.connected_components(layer_graph):
            layer_segments[layer].append(layer_graph.subgraph(c).copy())
    return layer_segments


def get_contour_points(segment, adj=8):
    """
    Find contour points of a segment.

    *********************************
    """
    contour = []
    for pt in segment.nodes:
        if segment.degree[pt] < adj:
            contour.append(pt)
    return contour


def convert_coord(coord, slide_dims):
    """
    Compute relative coords from abs.

    *********************************
    """
    sx, sy = slide_dims
    x, y = coord
    return float(x) / sx, float(y) / sy


def find_bottom_left(pts):
    """
    Find bottom left point.

    ***********************
    """
    ymax = max([pt[1] for pt in pts])
    xmin = 100000000000
    bl = None
    for pt in pts:
        x, y = pt
        if y == ymax:
            if x <= xmin:
                bl = x, y
    if bl is not None:
        return bl
    raise BottomLeftNotFound("Did not find bottom left point of the cloud !!!")


def turn_left(orientation):
    """
    Compute a new orientation after turning on the left.

    **************************************************
    """
    new_orientation = dict()
    new_orientation["front"] = orientation["left"]
    new_orientation["left"] = orientation["back"]
    new_orientation["back"] = orientation["right"]
    new_orientation["right"] = orientation["front"]
    return new_orientation


def turn_right(orientation):
    """
    Compute a new orientation after turning on the right.

    *****************************************************
    """
    new_orientation = dict()
    new_orientation["front"] = orientation["right"]
    new_orientation["left"] = orientation["front"]
    new_orientation["back"] = orientation["left"]
    new_orientation["right"] = orientation["back"]
    return new_orientation


def turn_back(orientation):
    """
    Compute a new orientation after turning back.

    *********************************************
    """
    # basically, it's just 'turn_right' twice...
    new_orientation = turn_right(orientation)
    new_new_orientation = turn_right(new_orientation)
    return new_new_orientation


def go_to_next_point(pt, orientation, perimeter):
    """
    Compute next point in the path.

    *******************************
    """
    x, y = pt
    front = x + orientation["front"][0], y + orientation["front"][1]
    left = x + orientation["left"][0], y + orientation["left"][1]
    right = x + orientation["right"][0], y + orientation["right"][1]
    back = x + orientation["back"][0], y + orientation["back"][1]
    if left in perimeter:
        new_orientation = turn_left(orientation)
        return left, new_orientation
    if front in perimeter:
        return front, orientation
    if right in perimeter:
        new_orientation = turn_right(orientation)
        return right, new_orientation
    if back in perimeter:
        new_orientation = turn_back(orientation)
        return back, new_orientation
    raise NextPointNotFound(
        "Point {} \nhas no next point in neighborhood {} \nthat is in perimeter {}".format(
            pt, {"left": left, "front": front, "right": right, "back": back}, perimeter
        )
    )


def compute_path(pts, d):
    """
    Compute the path around a segment.

    **********************************
    """
    path = []
    # first set remaining points to the whole cloud
    perimeter = set(pts)
    # find the bottom left point
    start_point = find_bottom_left(perimeter)
    path.append(start_point)
    # set the initial orientation
    start_orientation = {
        "front": (0, -d),
        "left": (-d, 0),
        "back": (0, d),
        "right": (d, 0)
    }
    current_point, orientation = go_to_next_point(
        start_point, start_orientation, perimeter
    )
    while current_point != start_point:
        path.append(current_point)
        # remaining.remove(current_point)
        next_point, next_orientation = go_to_next_point(
            current_point, orientation, perimeter
        )
        current_point = next_point
        orientation = next_orientation
    if len(path) > 0:
        return path
    else:
        raise NoPathFound(
            "No path found for {}, with interval {}!!!".format(pts, d)
        )


def layer_segment_to_json_struct(
    interval,
    layer_segments,
    layer_meta,
    slide
):
    """
    Create the json annotation file from the segments.

    **************************************************
    """
    slide_id = os.path.basename(slide._filename)
    # annotations = {"slide_id": slide_id, "layers": dict()}
    annotations = {"slide_id": slide_id, "layers": []}
    for layer, segments in layer_segments.items():
        meta = layer_meta[layer]
        layer_annotation = {
            "id": meta["label"],
            "color": meta["color"],
            "shapes": []
        }
        # annotations["layers"][layer] = dict()
        for idx, segment in enumerate(segments):
            # create one annotation by segment
            try:
                contour = get_contour_points(segment, adj=8)
                polygon = compute_path(contour, interval)
                polygon = [convert_coord(
                    pt, slide.dimensions
                ) for pt in polygon]
                shape = {
                    "points": [
                        {"x": x * 100, "y": y * 100,
                         "status": "written"} for x, y in polygon
                    ],
                    "id": str(idx),
                    "author": meta["author"],
                    "text": meta["text"],
                    "color": meta["color"],
                    "label": meta["label"],
                    "date": meta["date"]
                }
                layer_annotation["shapes"].append(shape)
            except (NoPathFound, NextPointNotFound, BottomLeftNotFound) as e:
                warnings.warn(str(e))
        annotations["layers"].append(layer_annotation)
    return annotations
