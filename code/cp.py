import xml.etree.ElementTree as ET
from svg.path import parse_path

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
def remove_namespace(element):
    for elem in element.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]  # remove the namespace
import svgwrite     
def double_points(svg_path, min_control_points):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    done = True  
    paths =  root.findall(".//path")
    if not paths:
        paths = root.findaa(".//{http://www.w3.org/2000/svg}path")
    for path in paths:
        d = path.get('d')
        new_d = ''
        commands = parse_path(d)
        print("initial length: ", len(commands))
        if(len(commands) > min_control_points):
            print("final length: ", len(commands))
            continue
        done = False
        for command in commands:
            command_name = str(command).split('(', 1)[0]
            if command_name in ["Move"]:
                new_d += f'M {command.start.real} {command.start.imag} '
            elif command_name in ["Line"]:
                starti = command.start.real
                startj = command.start.imag
                endi = command.end.real
                endj = command.end.imag
                midi = (starti + endi) / 2
                midj = (startj + endj) / 2
                new_d += f'L {midi} {midj} '
                new_d += f'L {endi} {endj} '
            elif command_name in ["CubicBezier"]:
                new_curve1, new_curve2 = split_cubic_bezier(command.start, command.control1, command.control2, command.end)
                new_d += f'C {new_curve1[1].real} {new_curve1[1].imag} {new_curve1[2].real} {new_curve1[2].imag} {new_curve1[3].real} {new_curve1[3].imag} '
                new_d += f'C {new_curve2[1].real} {new_curve2[1].imag} {new_curve2[2].real} {new_curve2[2].imag} {new_curve2[3].real} {new_curve2[3].imag} '
            elif command_name in ["QuadraticBezier"]:
                starti = command.start.real 
                startj = command.start.imag
                controli = command.control.real
                controlj = command.control.imag
                endi = command.end.real
                endj = command.end.imag
                midi = (starti + 2*controli + endi) / 4
                midj = (startj + 2*controlj + endj) / 4
                new_d += f'Q {(starti+ controli)/2} {(startj + controlj)/2} {midi} {midj} '
                new_d += f'Q {(controli + endi)/2} {(controlj + endj)/2} {endi} {endj} '
            
        path.set('d', new_d.strip())
        remove_namespace(root)

    tree.write(svg_path)
    return done

def process_svg(svg_path, min_control_points = 100):
    while(not double_points(svg_path, min_control_points)):
        pass