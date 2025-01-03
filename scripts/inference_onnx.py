import argparse
import onnxruntime
import numpy as np
import cv2
import time


def read_classnames(classname_path):
    with open(classname_path, "r") as f:
        classnames = f.read().splitlines()
        classnames = [name.strip() for name in classnames]
    return classnames


def preprocess_input(imgs: list[np.ndarray]):
    inps = [cv2.resize(img, (224, 224)) for img in imgs]
    inps = np.stack(inps, dtype=np.float32).transpose((3, 0, 1, 2))[None, ...]
    return inps


def rescale_bboxes(bboxes: np.ndarray, dest_width: int, dest_height: int):
    bboxes[..., [0, 2]] = np.clip(
        bboxes[..., [0, 2]] * dest_width, a_min=0., a_max=dest_width
    )
    bboxes[..., [1, 3]] = np.clip(
        bboxes[..., [1, 3]] * dest_height, a_min=0., a_max=dest_height
    )

    return bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="Input source (camera id or video file path)")
    parser.add_argument("--conf", default=0.6, type=float,
                        help="Confidence threshold")
    parser.add_argument("--len-clip", default=16, type=int,
                        help="Length of frame sequence for model")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Checkpoint path")
    parser.add_argument("--classname", type=str,
                        required=True, help="Class name")
    parser.add_argument("--multiclass", action="store_true",
                        default=False, help="Multiclass")
    args = parser.parse_args()

    CONF_THRESH = args.conf
    CLASSNAMES = read_classnames(args.classname)
    time_lst = []

    ort_session = onnxruntime.InferenceSession(args.checkpoint)
    input_names = ort_session.get_inputs()
    output_names = ort_session.get_outputs()

    input_names = [inp.name for inp in input_names]
    output_names = [out.name for out in output_names]

    confidence = np.array(args.conf, dtype=np.float64)

    frames = []
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        # print(height, width) # 240, 320

        if len(frames) <= 0:
            for _ in range(args.len_clip):
                frames.append(frame)

        frames.append(frame)
        del frames[0]

        inps = preprocess_input(frames)
        vis_frame = frames[-1].copy()

        ort_inputs = {
            input_names[0]: inps,
            input_names[1]: confidence
        }
        start = time.perf_counter()
        outputs = ort_session.run(
            output_names=output_names, input_feed=ort_inputs)[0]
        end = time.perf_counter() - start
        print("Inference time: ", end)
        time_lst.append(end)
        if not args.multiclass:
            labels = outputs[:, -1].astype(np.uint64)
            bboxes = rescale_bboxes(
                outputs[:, :4],
                dest_height=height,
                dest_width=width
            ).astype(np.uint32)
            scores = outputs[:, 4]
            if len(scores) > 0:
                for i in range(len(scores)):
                    cv2.rectangle(vis_frame, (bboxes[i][0], bboxes[i][1]),
                                  (bboxes[i][2], bboxes[i][3]), (0, 255, 0), 2)
                    y_coord = np.clip(bboxes[i][1], a_min=5, a_max=height)
                    cv2.putText(vis_frame, f"{CLASSNAMES[labels[i]]}: {scores[i]:.2f}", (
                        bboxes[i][0], y_coord - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            scores = outputs[:, 4:]
            bboxes = rescale_bboxes(
                outputs[:, :4],
                dest_height=height,
                dest_width=width
            ).astype(np.uint32)
            for i in range(scores.shape[0]):
                x1, y1, x2, y2 = bboxes[i]
                inds = np.where(scores[i] > confidence)[0]
                # draw bbox
                cv2.rectangle(vis_frame, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)
                # draw text
                blk = np.zeros(vis_frame.shape, np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text = []
                text_size = []
                for _, cls_ind in enumerate(inds):
                    text.append("[{:.2f}] ".format(
                        scores[i][cls_ind]) + str(CLASSNAMES[cls_ind]))
                    text_size.append(cv2.getTextSize(
                        text[-1], font, fontScale=0.5, thickness=1)[0])
                    coord.append((x1+3, y1+14+20*_))
                    cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-12), (coord[-1][0]+text_size[-1]
                                                                           [0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                vis_frame = cv2.addWeighted(vis_frame, 1.0, blk, 0.5, 1)
                for t in range(len(text)):
                    cv2.putText(
                        vis_frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)

        cv2.imshow("frame", vis_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Avg inference time: ", np.mean(time_lst))
