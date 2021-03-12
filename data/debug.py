import os
import os.path as osp

import cv2
import xml.etree.ElementTree as ET

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects

def draw_bbox(img_path, output_dir, is_whole_bbox=False):
  # Load the demo image
  im = cv2.imread(img_path)

  voc_dir = "WholeBBVOC" if is_whole_bbox else "ObjectsVOC"
  voc_file = img_path.replace("Images", voc_dir)
  voc_file = voc_file.replace(".jpg", ".xml")
  # Load the ground-truth bounding box
  # there is only one object of interest
  gt = parse_rec(voc_file)[0]
  #print(gt)
  # draw the gt bbox
  gt_bbox = gt["bbox"]
  gt_label = gt["name"]
  gt_color = (0, 0, 255)
  gt_thickness = 5
  im = cv2.rectangle(im, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]),
                  gt_color, gt_thickness)
  """
  im = cv2.putText(im, gt_label,
                  (gt_bbox[0], gt_bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                  gt_color, gt_thickness, bottomLeftOrigin=False)
  """
  output_path = img_path.replace("Images", output_dir)
  output_dirname = osp.dirname(output_path)
  if not osp.exists(output_dirname):
    os.makedirs(output_dirname)
  
  # write a debugging file
  print("Writing {}".format(output_path))
  cv2.imwrite(output_path, im)


if __name__ == "__main__":
  dataset = "TEgO"
  is_whole_bbox = True
  dataset_path = osp.join(dataset, "Training", "Images", "in-the-wild")
  # get a list of images
  img_list = [osp.join(dirname, filename) \
              for dirname, _, filenames in os.walk(dataset_path) \
                for filename in filenames]
  
  output_dir = "DebugWholeBbox" if is_whole_bbox else "DebugBbox"
  
  for each in img_list:
    #print(each)
    draw_bbox(each, output_dir, is_whole_bbox)

  