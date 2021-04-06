import argparse
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import tensorflow as tf

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.compat.v1.Session()
        with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name)
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images", output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=Path)
    parser.add_argument('detections_path', type=Path)
    parser.add_argument('model')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--save-tracks', default=False, action='store_true')
    args = parser.parse_args()

    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 1.0

    # Open the source video and read metadata
    video_capture = cv2.VideoCapture(str(args.video_path))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    video_writer = cv2.VideoWriter(str(Path(args.output_dir, 'video.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # Read detections from file
    with open(args.detections_path, 'r') as file:
        detections_json = json.load(file)

    # Save detections and tracks in JSON format. Each frame will have a list of detections and a list of tracks
    tracks_output_json = {}

    # Create encoder
    encoder = create_box_encoder(args.model, batch_size=32)

    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric, n_init=10, max_age=30)

    visualizer = visualization.Visualization({
        "sequence_name": 'customSequence',
        "image_size": frame_size,
        "min_frame_idx": None,
        "max_frame_idx": None,
    }, update_ms=1000/fps)

    # Process video frames
    for i in range(frame_count):

        # Show status
        print(f'Processing frame {i + 1}/{frame_count}')

        # Read frame
        success, frame = video_capture.read()
        if not success:
            print(f'[WARNING] Failed to read frame {i + 1}!', file=sys.stderr)
            continue

        # Process detections
        detections = []
        for d in detections_json[f'{i}']:  # The key is a string so we have to convert the index

            tlwh = (d['x1'], d['y1'], d['x2'] - d['x1'], d['y2'] - d['y1'])

            # Encode features
            features = encoder(frame, [tlwh]).squeeze()

            detections.append(Detection(tlwh, d['score'], features))

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker
        tracker.predict()
        tracker.update(detections)

        # Draw detections and tracks
        visualizer.set_image(frame)
        visualizer.draw_detections(detections)
        visualizer.draw_trackers(tracker.tracks)

        # Save results
        video_writer.write(frame)

        tracks = []
        for track in tracker.tracks:
            box = track.to_tlwh()
            tracks.append({
                'id': track.track_id,
                'x1': box[0],
                'y1': box[1],
                'x2': box[0] + box[2],
                'y2': box[1] + box[3]
            })

        tracks_output_json[i] = tracks

    # Close reader and writer
    video_writer.release()
    video_capture.release()

    # Save detections
    if args.save_tracks:
        with open(Path(args.output_dir, 'tracks.json'), 'w') as file:
            json.dump(tracks_output_json, file)
