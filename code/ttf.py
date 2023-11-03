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
import svgwrite
# import cairosvg
import aspose.words as aw
import aspose.pydrawing as pydraw



device = torch.device("cuda" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

reload(bezier)

def fix_single_svg(svg_path, all_word=False):
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

    w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max([torch.max(p.points[:, 0]) for p in shapes])
    h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max([torch.max(p.points[:, 1]) for p in shapes])

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] + target_canvas_width/2 - int(w_min + (w_max - w_min) / 2)
        p.points[:, 1] = p.points[:, 1] + target_canvas_height/2 - int(h_min + (h_max - h_min) / 2)

    output_path = f"{svg_path[:-4]}_scaled.svg"
    save_svg.save_svg(output_path, target_canvas_width, target_canvas_height, shapes, shape_groups)

def normalize_letter_size(dest_path, font, word, letter_index):
    fontname = os.path.splitext(os.path.basename(font))[0]
    fname = f"{dest_path}/{fontname}_{word}.svg"
    
    fname = fname.replace(" ", "_")
    fix_single_svg(fname, all_word=True)
    

def glyph_to_cubics(face, x=0, y=0):
    ''' Convert current font face glyph to cubic beziers'''

    def linear_to_cubic(Q):
        a, b = Q
        return [a + (b - a) * t for t in np.linspace(0, 1, 4)]

    def quadratic_to_cubic(Q):
        return [Q[0],
                Q[0] + (2 / 3) * (Q[1] - Q[0]),
                Q[2] + (2 / 3) * (Q[1] - Q[2]),
                Q[2]]

    beziers = []
    pt = lambda p: np.array([x + p.x, - p.y - y])  # Flipping here since freetype has y-up
    last = lambda: beziers[-1][-1]

    def move_to(a, beziers):
        beziers.append([pt(a)])

    def line_to(a, beziers):
        Q = linear_to_cubic([last(), pt(a)])
        beziers[-1] += Q[1:]

    def conic_to(a, b, beziers):
        Q = quadratic_to_cubic([last(), pt(a), pt(b)])
        beziers[-1] += Q[1:]

    def cubic_to(a, b, c, beziers):
        beziers[-1] += [pt(a), pt(b), pt(c)]

    face.glyph.outline.decompose(beziers, move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
    beziers = [np.array(C).astype(float) for C in beziers]
    return beziers

# def handle_ligature(glyph_infos, glyph_positions):
#     combined_advance = sum(pos.x_advance for pos in glyph_positions)
#     first_x_offset = glyph_positions[0].x_offset

#     combined_advance = x_adv_1 + x_adv_2




#     # Adjust the x_offset values based on the difference between the first glyph's x_offset and the combined_advance
#     for pos in glyph_positions:
#         pos.x_offset += combined_advance - pos.x_advance - first_x_offset

#     # Render the ligature using the adjusted glyph positions
#     render_glyphs(glyph_infos, glyph_positions)


def font_string_to_beziers(font, txt, size=30, spacing=1.0, merge=True, target_control=None):
    ''' Load a font and convert the outlines for a given string to cubic bezier curves,
        if merge is True, simply return a list of all bezier curves,
        otherwise return a list of lists with the bezier curves for each glyph'''
    print(font)
    
    vhb = hb.Vharfbuzz(font)
    buf = vhb.shape(txt, {"features": {"kern": True, "liga": True}})

    buf.guess_segment_properties()

    glyph_infos = buf.glyph_infos
    glyph_positions = buf.glyph_positions
    glyph_count = {glyph_infos[i].cluster: 0 for i in range(len(glyph_infos))}
    
    svg = vhb.buf_to_svg(buf)
    paths, attributes = svgstr2paths(svg)
    
    face = ft.Face(font)
    face.set_char_size(64 * size)
    pindex = -1

    x, y = 0, 0
    beziers, chars = [], []

    for path_idx, path in enumerate(paths):
        segment_vals = []
        print("="*20 + str(path_idx) + "="*20)
        for segment in path:
            segment_type = segment.__class__.__name__
            t_values = np.linspace(0, 1, 10)
            points = [segment.point(t) for t in t_values]
            for pt in points:
                segment_vals += [[pt.real, -pt.imag]]

            # points = [bezier.point(t) for t in t_values]

            if segment_type == 'Line':
                # Line segment
                start = segment.start
                end = segment.end
                print(f"Line: ({start.real}, {start.imag}) to ({end.real}, {end.imag})")
            
            elif segment_type == 'QuadraticBezier':
                # Quadratic Bézier segment
                start = segment.start
                control = segment.control
                end = segment.end
                print(f"Quadratic Bézier: ({start.real}, {start.imag}) to ({end.real}, {end.imag}) with control point ({control.real}, {control.imag})")
            
            elif segment_type == 'CubicBezier':
                # Cubic Bézier segment
                start = segment.start
                control1 = segment.control1
                control2 = segment.control2
                end = segment.end
                print(f"Cubic Bézier: ({start.real}, {start.imag}) to ({end.real}, {end.imag}) with control points ({control1.real}, {control1.imag}) and ({control2.real}, {control2.imag})")
            
            else:
                # Other segment types (Arc, Close)
                print(f"Segment type: {segment_type}")

        beziers += [[np.array(segment_vals)]]

    beziers_2 = []
    glyph_infos = glyph_infos[::-1]
    glyph_positions = glyph_positions[::-1]
    for i, (info, pos) in enumerate(zip(glyph_infos, glyph_positions)):
        index = info.cluster
        c = f"{txt[index]}_{glyph_count[index]}"
        chars += [c]
        glyph_count[index] += 1
        glyph_index = info.codepoint
        face.load_glyph(glyph_index, flags=ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP)
        # face.load_char(c, ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP)

        findex = -1 
        if i+1 < len(glyph_infos):
            findex = glyph_infos[i+1].cluster
            foffset = (glyph_positions[i+1].x_offset, glyph_positions[i+1].y_offset)
            fadvance = (glyph_positions[i+1].x_advance, glyph_positions[i+1].y_advance)
        
        # bez = glyph_to_cubics(face, x+pos.x_offset+pos.x_advance, y+pos.y_offset+pos.y_advance)
        # if findex != index:
        #     x += pos.x_offset
        #     y += pos.y_offset
        # else:
        #     x += pos.x_offset
        #     y += pos.y_offset


        bez = glyph_to_cubics(face, x, y)
            

        # Check number of control points if desired
        if target_control is not None:
            if c in target_control.keys():
                nctrl = np.sum([len(C) for C in bez])
                while nctrl < target_control[c]:
                    longest = np.max(
                        sum([[bezier.approx_arc_length(b) for b in bezier.chain_to_beziers(C)] for C in bez], []))
                    thresh = longest * 0.5
                    bez = [bezier.subdivide_bezier_chain(C, thresh) for C in bez]
                    nctrl = np.sum([len(C) for C in bez])
                    print(nctrl)

        if merge:
            beziers_2 += bez
        else:
            beziers_2.append(bez)

        # kerning = face.get_kerning(index, findex)
        # x += (slot.advance.x + kerning.x) * spacing
        # previous = txt[index]
    
        # print(f"C: {txt[index]}/{index} | X: {x+pos.x_offset}| Y: {y+pos.y_offset}")
        print(f"C: {txt[index]}/{index} | X: {x}: {pos.x_advance}/{pos.x_offset} | Y: {y}: {pos.y_advance}/{pos.y_offset}")
        
        # if findex != index:
        x -= pos.x_advance
        # y += pos.y_advance + pos.y_offset

        pindex = index
            
    return beziers_2, chars


def bezier_chain_to_commands(C, closed=True):
    curves = bezier.chain_to_beziers(C)
    cmds = 'M %f %f ' % (C[0][0], C[0][1])
    n = len(curves)
    for i, bez in enumerate(curves):
        if i == n - 1 and closed:
            cmds += 'C %f %f %f %f %f %fz ' % (*bez[1], *bez[2], *bez[3])
        else:
            cmds += 'C %f %f %f %f %f %f ' % (*bez[1], *bez[2], *bez[3])
    return cmds


def count_cp(file_name, font_name):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_name)
    p_counter = 0
    for path in shapes:
        p_counter += path.points.shape[0]
    print(f"TOTAL CP:   [{p_counter}]")
    return p_counter


def write_letter_svg(c, header, fontname, beziers, subdivision_thresh, dest_path):
    cmds = ''
    svg = header

    path = '<g><path d="'
    for C in beziers:
        if subdivision_thresh is not None:
            print('subd')
            C = bezier.subdivide_bezier_chain(C, subdivision_thresh)
        cmds += bezier_chain_to_commands(C, True)
    path += cmds + '"/>\n'
    svg += path + '</g></svg>\n'

    fname = f"{dest_path}/{fontname}_{c}.svg"
    fname = fname.replace(" ", "_")
    f = open(fname, 'w')
    f.write(svg)
    f.close()
    return fname, path

def write_letter_svg_hb(vhb, c, dest_path, fontname):
    buf = vhb.shape(c, {"features": {"kern": True, "liga": True}})    
    svg = vhb.buf_to_svg(buf)

    fname = f"{dest_path}/{fontname}_{c}.svg"
    fname = fname.replace(" ", "_")
    f = open(fname, 'w')
    f.write(svg)
    f.close()
    return fname

def font_string_to_svgs(dest_path, font, txt, size=30, spacing=1.0, target_control=None, subdivision_thresh=None):

    fontname = os.path.splitext(os.path.basename(font))[0]
    glyph_beziers, chars = font_string_to_beziers(font, txt, size, spacing, merge=False, target_control=target_control)
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    # Compute boundig box
    points = np.vstack(sum(glyph_beziers, []))
    lt = np.min(points, axis=0)
    rb = np.max(points, axis=0)
    size = rb - lt

    sizestr = 'width="%.1f" height="%.1f"' % (size[0], size[1])
    boxstr = ' viewBox="%.1f %.1f %.1f %.1f"' % (lt[0], lt[1], size[0], size[1])
    header = '''<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" baseProfile="full" '''
    header += sizestr
    header += boxstr
    header += '>\n<defs/>\n'

    svg_all = header

    print(f"Len Glyph Bezier: {len(glyph_beziers)} | Chars: {len(chars)}")
    for i, (c, beziers) in enumerate(zip(chars, glyph_beziers)):
        print(f"==== {c} ====")
        fname, path = write_letter_svg(c, header, fontname, beziers, subdivision_thresh, dest_path)

        num_cp = count_cp(fname, fontname)
        print(num_cp)
        print(font, c)
        # Add to global svg
        svg_all += path + '</g>\n'

    vhb = hb.Vharfbuzz(font)
    buf = vhb.shape(txt, {"features": {"kern": True, "liga": True}})    
    svg = vhb.buf_to_svg(buf)

    # Save global svg
    svg_all += '</svg>\n'
    fname = f"{dest_path}/{fontname}_{txt}.svg"
    fname = fname.replace(" ", "_")
    f = open(fname, 'w')
    f.write(svg)
    f.close()
    return chars

def font_string_to_svgs_hb(dest_path, font, txt, size=30, spacing=1.0, target_control=None, subdivision_thresh=None):

    fontname = os.path.splitext(os.path.basename(font))[0]
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    vhb = hb.Vharfbuzz(font)
    buf = vhb.shape(txt, {"features": {"kern": True, "liga": True}})
    buf.guess_segment_properties()

    buf = vhb.shape(txt, {"features": {"kern": True, "liga": True}})    
    svg = vhb.buf_to_svg(buf)

    # Save global svg
    fname = f"{dest_path}/{fontname}_{txt}.svg"
    fname = fname.replace(" ", "_")
    f = open(fname, 'w')
    f.write(svg)
    f.close()
    return None

def extract_attributes(input_svg_file):
    tree = ET.parse(input_svg_file)
    root = tree.getroot()

    xmlns = root.attrib.get('xmlns', '')
    width = root.attrib.get('width', '')
    height = root.attrib.get('height', '')

    return xmlns, width, height
def remove_namespace(element, namespace):
    for elem in element.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]  # remove the namespace

def extract_svg_paths(dest_path, font, word, letter_index):
    fontname = os.path.splitext(os.path.basename(font))[0]
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    word_svg = f"{dest_path}/{fontname}_{word}_scaled.svg"
    tree = ET.parse(word_svg)
    root = tree.getroot()
    xmlns, width, height = extract_attributes(word_svg)

    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    letter_path = paths[len(paths) - letter_index - 1]
    new_root = ET.Element("svg", xmlns=xmlns, width=width, height=height)
    new_root.append(letter_path)
    remove_namespace(new_root, xmlns)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(f"{dest_path}/{fontname}_{word}_{word[letter_index]}_scaled.svg")
        
def combine_word_mod(word, letter, font, experiment_dir):
    word_svg_scaled = f"./code/data/init/{font}_{word}_scaled.svg"
    svg_result = os.path.join(experiment_dir, "output-svg", "output.svg")

    
    tree = ET.parse(word_svg_scaled)
    root = tree.getroot()
    xmlns, width, height = extract_attributes(word_svg_scaled)
    namespace = {'svg': 'http://www.w3.org/2000/svg'}
    paths = root.findall(".//svg:path", namespaces=namespace)
    new_root = ET.Element("svg", xmlns=xmlns, width=width, height=height)
    
    letter_tree = ET.parse(svg_result)
    letter_root = letter_tree.getroot()
    letter_path = letter_root.findall(".//svg:path", namespaces=namespace)

    for i, path in enumerate(paths):
        if i == (len(paths) - letter - 1):
            new_root.append(letter_path[0])
        else:
            new_root.append(path)
    remove_namespace(new_root, xmlns)
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

if __name__ == '__main__':

    fonts = ["KaushanScript-Regular"]
    level_of_cc = 1

    if level_of_cc == 0:
        target_cp = None

    else:
        target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                     "E": 120, "F": 120, "G": 120, "H": 120,
                     "I": 35, "J": 80, "K": 100, "L": 80,
                     "M": 100, "N": 100, "O": 100, "P": 120,
                     "Q": 120, "R": 130, "S": 110, "T": 90,
                     "U": 100, "V": 100, "W": 100, "X": 130,
                     "Y": 120, "Z": 120,
                     "a": 120, "b": 120, "c": 100, "d": 100,
                     "e": 120, "f": 120, "g": 120, "h": 120,
                     "i": 35, "j": 80, "k": 100, "l": 80,
                     "m": 100, "n": 100, "o": 100, "p": 120,
                     "q": 120, "r": 130, "s": 110, "t": 90,
                     "u": 100, "v": 100, "w": 100, "x": 130,
                     "y": 120, "z": 120
                     }

        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    for f in fonts:
        print(f"======= {f} =======")
        font_path = f"data/fonts/{f}.ttf"
        output_path = f"data/init"
        txt = "BUNNY"
        subdivision_thresh = None
        font_string_to_svgs(output_path, font_path, txt, target_control=target_cp,
                            subdivision_thresh=subdivision_thresh)
        normalize_letter_size(output_path, font_path, txt)

        print("DONE")



