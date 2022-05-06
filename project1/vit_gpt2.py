from argparse import ArgumentParser

from PIL import Image
import matplotlib.pyplot as plt
import requests
import torch
from transformers import (AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor)


def get_args():

    ap = ArgumentParser(description='args for comp329 image caption')
    ap.add_argument('-i','--input', type=str)
    ap.add_argument('-t','--trial', action='store_true')

    args = ap.parse_args()

    if not args.input.split('.')[-1].lower() in ['jpg','jpeg']:
        print('only processes jpg images')
        quit()

    return args


def load_pretrained():
    loc = "ydshieh/vit-gpt2-coco-en"

    feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
    tokenizer = AutoTokenizer.from_pretrained(loc)
    model = VisionEncoderDecoderModel.from_pretrained(loc)
    model.eval()

    return feature_extractor, tokenizer, model


def predict(image):

    feature_extractor, tokenizer, model = load_pretrained()
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds


def display(image,preds):
    plt.imshow(image)
    plt.title(preds)
    plt.show()


def main():

    args = get_args()

    if args.trial or ( not args.input ):
        # We will verify our results on an image of cute cats
        # should produce
        # ['a cat laying on top of a couch next to another cat']

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        with Image.open(requests.get(url, stream=True).raw) as image:
            preds = predict(image)

        display(image,preds)
        quit()

    image = Image.open(args.input)
    preds = predict(image)
    display(image,preds)


if __name__ == '__main__':
    main()
