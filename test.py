import re
import torch
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

class BoxQuantizer(object):
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            quantized_xmin = (
                xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (
                ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (
                xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (
                ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        return quantized_boxes

    def dequantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_xmin = (xmin + 0.5) * size_per_bin_w
            dequantized_ymin = (ymin + 0.5) * size_per_bin_h
            dequantized_xmax = (xmax + 0.5) * size_per_bin_w
            dequantized_ymax = (ymax + 0.5) * size_per_bin_h

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        dequantized_boxes = torch.cat(
            (dequantized_xmin, dequantized_ymin,
             dequantized_xmax, dequantized_ymax), dim=-1
        )

        return dequantized_boxes

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6"
)

box_quantizer = processor.post_processor.box_quantizer

text = "candy<loc_10><loc_20><loc_30><loc_40>candy<loc_5><loc_6><loc_7><loc_8>."
# pattern = r'([a-zA-Z0-9 ]+)<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>'
image_size = (1000, 1000)
allow_empty_phrase = True

text = text.replace('<s>', '')
text = text.replace('</s>', '')
text = text.replace('<pad>', '')

if allow_empty_phrase:
    pattern = rf"(?:(?:<loc_\d+>){{4,}})"
else:
    pattern = r"([^<]+(?:<loc_\d+>){4,})"
phrases = re.findall(pattern, text)
print(phrases)

# pattern should be text pattern and od pattern
pattern = r'^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)'
box_pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'

instances = []
for pharse_text in phrases:
    phrase_text_strip = pharse_text.replace('<ground>', '', 1)
    phrase_text_strip = pharse_text.replace('<obj>', '', 1)

    if phrase_text_strip == '' and not allow_empty_phrase:
        continue

    # parse phrase, get string 
    print(phrase_text_strip)
    phrase = re.search(pattern, phrase_text_strip)
    print(phrase)
    if phrase is None:
        continue

    phrase = phrase.group()
    # remove leading and trailing spaces
    phrase = phrase.strip()

    # parse bboxes by box_pattern
    bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
    if len(bboxes_parsed) == 0:
        continue

    # a list of list 
    bbox_bins = [[int(_bboxes_parsed.group(j)) for j in range(1, 5)] for _bboxes_parsed in bboxes_parsed]

    bboxes = box_quantizer.dequantize(
        boxes=torch.tensor(bbox_bins),
        size=image_size
    ).tolist()  

    phrase = phrase.encode('ascii',errors='ignore').decode('ascii')
    for _bboxes in bboxes:
        # Prepare instance.
        instance = {}
        instance['bbox'] = _bboxes
        # exclude non-ascii characters
        instance['cat_name'] = phrase
        instances.append(instance)

print(instances)