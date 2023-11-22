from importlib import reload
import os
import numpy as np
import bezier
import freetype as ft
import pydiffvg
import torch
import save_svg
import vharfbuzz as hb
from svgpathtools import svgstr2paths
import xml.etree.ElementTree as ET
import aspose.words as aw
import aspose.pydrawing as pydraw
import wandb
import config as cfg
import xml.etree.ElementTree as ET
from svg.path import parse_path
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
)
reload(bezier)


def fix_single_svg(svg_path, output_path, all_word=False):
    target_h_letter = 360
    target_canvas_width, target_canvas_height = 600, 600
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    letter_h = canvas_height
    letter_w = canvas_width
    if all_word:
        if letter_w > letter_h:
            scale_canvas_w = target_h_letter / letter_w
            hsize = int(letter_h * scale_canvas_w)
            scale_canvas_h = hsize / letter_h
        else:
            scale_canvas_h = target_h_letter / letter_h
            wsize = int(letter_w * scale_canvas_h)
            scale_canvas_w = wsize / letter_w
    else:
        scale_canvas_h = target_h_letter / letter_h
        wsize = int(letter_w * scale_canvas_h)
        scale_canvas_w = wsize / letter_w
    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h + target_h_letter
        p.points[:, 1] = -p.points[:, 1]
        # p.points[:, 0] = -p.points[:, 0]
    w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
        [torch.max(p.points[:, 0]) for p in shapes]
    )
    h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
        [torch.max(p.points[:, 1]) for p in shapes]
    )
    for num, p in enumerate(shapes):
        p.points[:, 0] = (
            p.points[:, 0] + target_canvas_width / 2 - int(w_min + (w_max - w_min) / 2)
        )
        p.points[:, 1] = (
            p.points[:, 1] + target_canvas_height / 2 - int(h_min + (h_max - h_min) / 2)
        )
    save_svg.save_svg(
        output_path, target_canvas_width, target_canvas_height, shapes, shape_groups
    )


def normalize_svg_size(input_path):
    fname = f"{input_path}.svg"
    fname = fname.replace(" ", "_")
    target = f"{input_path}_scaled.svg"
    fix_single_svg(fname, target, all_word=True)


def remove_namespace(element):
    for elem in element.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]  # remove the namespace


def path_to_cubics(commands):
    def approximate_line(start, end):
        return [(start + 2 * end) / 3, (2 * start + end) / 3]

    segment_vals = ""
    for segment in commands:
        segment_type = segment.__class__.__name__
        if segment_type == "Move":
            segment_vals += f"M {segment.start.real} {segment.start.imag} "
        if segment_type == "Line":
            start = segment.start
            end = segment.end
            control_points = approximate_line(start, end)
            segment_vals += f"C {control_points[0].real} {control_points[0].imag} {control_points[1].real} {control_points[1].imag} {end.real} {end.imag} "
        elif segment_type == "QuadraticBezier":
            start = segment.start
            control = segment.control
            end = segment.end
            control1 = (2 / 3) * control + (1 / 3) * start
            control2 = (2 / 3) * control + (1 / 3) * end
            segment_vals += f"C {control1.real} {control1.imag} {control2.real} {control2.imag} {end.real} {end.imag} "
        elif segment_type == "CubicBezier":
            start = segment.start
            control1 = segment.control1
            control2 = segment.control2
            end = segment.end
            segment_vals += f"C {control1.real} {control1.imag} {control2.real} {control2.imag} {end.real} {end.imag} "
    return segment_vals


def split_cubic_bezier(start, control1, control2, end):
    mid1 = (start + control1) / 2
    mid2 = (control1 + control2) / 2
    mid3 = (control2 + end) / 2
    mid4 = (mid1 + mid2) / 2
    mid5 = (mid2 + mid3) / 2
    split_point = (mid4 + mid5) / 2
    new_curve1 = (start, mid1, mid4, split_point)
    new_curve2 = (split_point, mid5, mid3, end)
    return new_curve1, new_curve2


def increase_cp(commands, thresh):
    segment_vals = ""
    for segment in commands:
        segment_type = segment.__class__.__name__
        if segment_type == "Move":
            segment_vals += f"M {segment.start.real} {segment.start.imag} "
        elif segment_type == "CubicBezier":
            if segment.length() >= thresh:
                new_curve1, new_curve2 = split_cubic_bezier(
                    segment.start, segment.control1, segment.control2, segment.end
                )
                segment_vals += f"C {new_curve1[1].real} {new_curve1[1].imag} {new_curve1[2].real} {new_curve1[2].imag} {new_curve1[3].real} {new_curve1[3].imag} "
                segment_vals += f"C {new_curve2[1].real} {new_curve2[1].imag} {new_curve2[2].real} {new_curve2[2].imag} {new_curve2[3].real} {new_curve2[3].imag} "
            else:
                start = segment.start
                control1 = segment.control1
                control2 = segment.control2
                end = segment.end
                segment_vals += f"C {control1.real} {control1.imag} {control2.real} {control2.imag} {end.real} {end.imag} "
    return segment_vals


def font_string_to_beziers(svg_path, txt, target_control=None):
    """Load a font and convert the outlines for a given string to cubic bezier curves,
    if merge is True, simply return a list of all bezier curves,
    otherwise return a list of lists with the bezier curves for each glyph"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    done = True
    paths = root.findall(".//path")
    if not paths:
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    for i, path in enumerate(paths):
        d = path.get("d")
        letter = txt[-(i + 1)]
        commands = parse_path(d)
        new_d = path_to_cubics(commands)
        # Check number of control points if desired
        if target_control is not None:
            if letter in target_control.keys():
                commands = parse_path(new_d)
                nctrl = len(commands)
                print("letter", letter)
                print("target_control[letter]", target_control[letter])
                print("nctrl", nctrl)
                while nctrl < target_control[letter]:
                    longest = np.max([segment.length() for segment in commands])
                    thresh = longest * 0.5
                    new_d = increase_cp(commands, thresh=thresh)
                    commands = parse_path(new_d)
                    nctrl = len(commands)
        path.set("d", new_d.strip())
        remove_namespace(root)
    tree.write(svg_path)


def font_string_to_svgs(
    target_path,
    font_path,
    txt,
    size=30,
    spacing=1.0,
    target_control=None,
    subdivision_thresh=None,
):
    vhb = hb.Vharfbuzz(font_path)
    buf = vhb.shape(txt, {"features": {"kern": True, "liga": True}})
    svg = vhb.buf_to_svg(buf)
    # svg_path = f"{dest_path}/{fontname}_{txt}.svg"
    # svg_path = svg_path.replace(" ", "_")
    target_path = f"{target_path}.svg"
    print("svg_path", target_path)
    f = open(target_path, "w")
    f.write(svg)
    f.close()
    font_string_to_beziers(target_path, txt, target_control=target_control)
    return


def extract_attributes(input_svg_file):
    tree = ET.parse(input_svg_file)
    root = tree.getroot()
    xmlns = root.attrib.get("xmlns", "")
    width = root.attrib.get("width", "")
    height = root.attrib.get("height", "")
    return xmlns, width, height


def extract_svg_paths(dest_path, letters):
    word_svg_path = f"{dest_path}_scaled.svg"
    tree = ET.parse(word_svg_path)
    root = tree.getroot()
    xmlns, width, height = extract_attributes(word_svg_path)
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    new_root = ET.Element("svg", xmlns=xmlns, width=width, height=height)
    for letter_idx in letters:
        letter_path = paths[len(paths) - int(letter_idx) - 1]
        new_root.append(letter_path)
    remove_namespace(new_root)
    new_tree = ET.ElementTree(new_root)
    letter_name = "".join([str(elem) for elem in letters])
    new_tree.write(word_svg_path)


def combine_word_mod(svg_path, word, letter, font, experiment_dir):
    word_svg_scaled = f"{svg_path}.svg"
    svg_path = svg_path[:-7]
    word_svg = f"{svg_path}.svg"
    normalize_svg_size(svg_path)
    word_svg_scaled = f"{svg_path}_scaled.svg"
    print(word_svg_scaled)
    svg_result = os.path.join(experiment_dir, "output-svg", "output.svg")

    tree = ET.parse(word_svg_scaled)
    root = tree.getroot()
    xmlns, width, height = extract_attributes(word_svg_scaled)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall(".//svg:path", namespaces=namespace)
    new_root = ET.Element("svg", xmlns=xmlns, width=width, height=height)
    letter_tree = ET.parse(svg_result)
    letter_root = letter_tree.getroot()
    letter_path = letter_root.findall(".//svg:path", namespaces=namespace)
    j = 0
    for i, path in enumerate(paths):
        print("path:", path)
        if abs(i - len(paths) + 1) in letter:
            new_root.append(letter_path[j])
            j += 1
        else:
            new_root.append(path)
    remove_namespace(new_root)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(f"{experiment_dir}/{font}_{word}_{letter}.svg")
    # Create and save a simple document
    doc = aw.Document()
    builder = aw.DocumentBuilder(doc)
    shape = builder.insert_image(f"{experiment_dir}/{font}_{word}_{letter}.svg")
    shape.fill.back_color = pydraw.Color.white
    pageSetup = builder.page_setup
    pageSetup.page_width = shape.width
    pageSetup.page_height = shape.height
    pageSetup.top_margin = 0
    pageSetup.left_margin = 0
    pageSetup.bottom_margin = 0
    pageSetup.right_margin = 0
    doc.save(f"{experiment_dir}/{font}_{word}_{letter}.png")
    if cfg.use_wandb:
        wandb.log(
            {
                "img": wandb.Image(
                    plt.imread(f"{experiment_dir}/{font}_{word}_{letter}.png")
                )
            } , step = 501
        )
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             "Compined": wandb.Image(
    #                 plt.imread(f"{experiment_dir}/{font}_{word}_{letter}.png")
    #             )
    #         }
    #     )


if __name__ == "__main__":
    fonts = ["ArefRuqaa-Regular"]
    level_of_cc = 1

    if level_of_cc == 0:
        target_cp = None

    else:
        target_cp = {
            "A": 120,
            "B": 120,
            "C": 100,
            "D": 100,
            "E": 120,
            "F": 120,
            "G": 120,
            "H": 120,
            "I": 35,
            "J": 80,
            "K": 100,
            "L": 80,
            "M": 100,
            "N": 100,
            "O": 100,
            "P": 120,
            "Q": 120,
            "R": 130,
            "S": 110,
            "T": 90,
            "U": 100,
            "V": 100,
            "W": 100,
            "X": 130,
            "Y": 120,
            "Z": 120,
            "a": 120,
            "b": 120,
            "c": 100,
            "d": 100,
            "e": 120,
            "f": 120,
            "g": 120,
            "h": 120,
            "i": 35,
            "j": 80,
            "k": 100,
            "l": 80,
            "m": 100,
            "n": 100,
            "o": 100,
            "p": 120,
            "q": 120,
            "r": 130,
            "s": 110,
            "t": 90,
            "u": 100,
            "v": 100,
            "w": 100,
            "x": 130,
            "y": 120,
            "z": 120,
        }

        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    for f in fonts:
        print(f"======= {f} =======")

        font_path = f"data/fonts/arabic/{f}.ttf"
        output_path = f"data/init"
        txt = "حصان"
        subdivision_thresh = None
        font_string_to_svgs(
            output_path,
            font_path,
            txt,
            target_control=target_cp,
            subdivision_thresh=subdivision_thresh,
        )
        # normalize_letter_size(output_path, font_path, txt)

        print("DONE")
    # combine_word_mod("/home/alaa/projects/me/latest/Font-To-Sketch/code/data/init/06_موسيقى_01_scaled", "موسيقى" , [0,1], "06", "/home/alaa/projects/me/latest/Font-To-Sketch/output/arabic/06_موسيقى_01_dot_loss_0.2_content_loss0_angels_loss0.5_seed_42" )
    # fonts = ["ArefRuqaa-Regular"]
    # level_of_cc = 1

    # if level_of_cc == 0:
    #     target_cp = None

    # else:
    #     target_cp = {
    #         "A": 120,
    #         "B": 120,
    #         "C": 100,
    #         "D": 100,
    #         "E": 120,
    #         "F": 120,
    #         "G": 120,
    #         "H": 120,
    #         "I": 35,
    #         "J": 80,
    #         "K": 100,
    #         "L": 80,
    #         "M": 100,
    #         "N": 100,
    #         "O": 100,
    #         "P": 120,
    #         "Q": 120,
    #         "R": 130,
    #         "S": 110,
    #         "T": 90,
    #         "U": 100,
    #         "V": 100,
    #         "W": 100,
    #         "X": 130,
    #         "Y": 120,
    #         "Z": 120,
    #         "a": 120,
    #         "b": 120,
    #         "c": 100,
    #         "d": 100,
    #         "e": 120,
    #         "f": 120,
    #         "g": 120,
    #         "h": 120,
    #         "i": 35,
    #         "j": 80,
    #         "k": 100,
    #         "l": 80,
    #         "m": 100,
    #         "n": 100,
    #         "o": 100,
    #         "p": 120,
    #         "q": 120,
    #         "r": 130,
    #         "s": 110,
    #         "t": 90,
    #         "u": 100,
    #         "v": 100,
    #         "w": 100,
    #         "x": 130,
    #         "y": 120,
    #         "z": 120,
    #     }

    #     target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    # for f in fonts:
    #     print(f"======= {f} =======")

    #     font_path = f"data/fonts/arabic/{f}.ttf"
    #     output_path = f"data/init"
    #     txt = "حصان"
    #     subdivision_thresh = None
    #     font_string_to_svgs(
    #         output_path,
    #         font_path,
    #         txt,
    #         target_control=target_cp,
    #         subdivision_thresh=subdivision_thresh,
    #     )
    #     # normalize_letter_size(output_path, font_path, txt)

    #     print("DONE")
