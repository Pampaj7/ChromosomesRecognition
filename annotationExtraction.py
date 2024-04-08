# Function to parse and extract annotation details from XML file
#XML is hierarchically structured data, so we need to parse it to extract the information we need :(
import xml.etree.ElementTree as ET


def extract_annotations(element):
    annotations = []
    for obj in element.findall('object'):
        annotation = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text,
            'truncated': obj.find('truncated').text,
            'difficult': obj.find('difficult').text,
            'bndbox': {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
        }
        annotations.append(annotation)
    return annotations


def extract_annotations_from_xml(xml_file):
    tree = ET.parse(xml_file)  # crazy shit by gpt
    root = tree.getroot()
    return extract_annotations(root)


