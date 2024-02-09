import os
import cv2
import numpy as np
import BboxToolkit as bt
from mmdet.datasets.pipelines.obb.misc import polymask2obb
import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES

def visualize_with_obboxes(img_path, obboxes, labels, args):
    """
    Visualize oriented bounding boxes on the image based on given arguments and save the visualization.

    Args:
        img_path (str): Path to the image file.
        obboxes (np.ndarray): Array of obboxes with shape [N, 5] where 5 -> (x_ctr, y_ctr, w, h, angle).
        labels (np.ndarray): Array of labels for the obboxes.
        args (dict): Dictionary of visualization arguments.
    """

    # Ensure the save directory exists
    save_path = args.get('save_path', '.')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load class names if provided
    class_names = None
    if args.get('shown_names') and os.path.isfile(args['shown_names']):
        with open(args['shown_names'], 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    # Convert obboxes to the specified bbox type if needed
    shown_btype = args.get('shown_btype')
    if shown_btype:
        obboxes = bt.bbox2type(obboxes, shown_btype)

    # Filtering by score threshold if scores are provided
    score_thr = args.get('score_thr', 0.2)
    if obboxes.shape[1] == 6:  # Assuming scores are provided as the last column
        scores = obboxes[:, 5]
        valid_indices = scores > score_thr
        obboxes = obboxes[valid_indices, :5]  # Exclude scores from obboxes
        labels = labels[valid_indices]
    else:
        scores = None

    # Visualization parameters
    colors = args.get('colors', 'green')
    thickness = args.get('thickness', 2.0)
    text_off = args.get('text_off', False)
    font_size = args.get('font_size', 10)

    # Call visualization function from BboxToolkit
    bt.imshow_bboxes(img_path, obboxes, labels=labels, scores=scores,
                     class_names=class_names, colors=colors, thickness=thickness,
                     with_text=not text_off, font_size=font_size, show=False,
                     wait_time=args.get('wait_time', 0), out_file=save_path)

args = {
    "save_dir": "",
    "save_path": "",
    "shown_btype": None,
    "shown_names": 
    "/home/liyuqiu/RS-PCT/thirdparty/OBBDetection/BboxToolkit/tools/vis_configs/dota1_5/short_names.txt",
    "score_thr": 0.3,
    "colors": 
    "/home/liyuqiu/RS-PCT/thirdparty/OBBDetection/BboxToolkit/tools/vis_configs/dota1_5/colors.txt",
    "thickness": 2.5,
    "text_off": False,
    "font_size": 12,
    "wait_time": 0
}

@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms, visualize=False):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        self.visualize = visualize
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        if self.visualize:
            parent_folder = "/home/liyuqiu/RS-PCT/data/DOTA/show_data_augs"
            base_name = os.path.splitext(data['img_info']['filename'])[0]
            folder_path = os.path.join(parent_folder, base_name)
            os.makedirs(folder_path, exist_ok=True)
            args['save_dir'] = folder_path
        for idx, t in enumerate(self.transforms):
            data = t(data)
            if data is None:
                return None
            
            if self.visualize:
                # Get the class name of the transform
                transform_class_name = type(t).__name__
                
                # Update args for the transformed image save path
                args['save_path'] = os.path.join(folder_path, 
                                                f'transformed_image_step_{idx}_{transform_class_name}.png')
                if 'gt_masks' in data and type(data['gt_masks']).__name__ != 'BitmapMasks':
                    # After applying each transform, visualize the image with the current state of GT obboxes
                    if type(data['img']).__name__ == 'DataContainer':
                        transformed_img = data['img'].data.clone().detach().numpy()
                    else:
                        transformed_img = data['img'].copy()
                    
                    if type(data['gt_masks']).__name__ == 'DataContainer':
                        transformed_masks = data['gt_masks'].data
                    else:
                        transformed_masks = data['gt_masks']
                    
                    transformed_obboxes = polymask2obb(transformed_masks)
                    if 'gt_labels' in data:
                        if type(data['gt_labels']).__name__ == 'DataContainer':
                            labels = data['gt_labels'].data.clone().detach().numpy()
                        else:
                            labels = data['gt_labels']
                        visualize_with_obboxes(transformed_img, transformed_obboxes, labels, args)
                    if 'transform_matrix' in data:
                        if type(data['img']).__name__ == 'DataContainer':
                            matrix_img = data['img'].data.clone().detach().numpy()
                        else:
                            matrix_img = data['img'].copy()
                        # Process each bbox
                        for bbox in data['ann_info']['bboxes']:
                            points = np.array([[bbox[i], bbox[i+1]] for i in range(0, len(bbox), 2)])
                            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))]) # Convert to homogeneous coordinates
                            transformed_points = (data['transform_matrix'] @ points_homogeneous.T).T

                            # Assuming the transformation doesn't include rotation, scale, or shear that would invalidate using cv2.polylines
                            # If it does, further processing to correctly draw the OBBs might be required
                            transformed_points = transformed_points[:, :2].astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(matrix_img, [transformed_points], isClosed=True, color=(0, 255, 0), thickness=2)
                        # Save or display the image
                        save_path = os.path.join(folder_path, 
                                                f'matrix_step_{idx}_{transform_class_name}.png')
                        cv2.imwrite(save_path, matrix_img)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
