import os
import time
import argparse
import imageio as iio
from PIL import Image
import matplotlib.pyplot as plt
from pylab import *
from gtts import gTTS
import pytesseract
from scipy import ndimage
from playsound import playsound


def process_text(text):
    adjusted = re.sub(r'[^a-zA-Z\s]', '', text)
    return adjusted.strip().lower()


def process_image(screenshot):
    img = screenshot.convert('L').transpose(Image.ROTATE_180)

    # To ndarray
    im = array(img)

    # To binary, make text (black) True
    im = (im < 40)

    # Do erode
    im = ndimage.binary_erosion(im, iterations=2).astype(im.dtype)

    # Invert to make text white
    im = ~im

    # Fill upper-half with zeros
    im[0:240, :] = True

    return Image.fromarray(im)


def test(save_image: bool):
    file_paths = []
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    for file_path in file_paths:
        screenshot = Image.open(file_path)

        img = process_image(screenshot)

        text = process_text(pytesseract.image_to_string(img, lang='eng'))

        filename = file_path.split('/')[-1].split('.')[0]

        if save_image and filename != text:
            img.save(os.path.join('data', 'failed', f'{filename}.jpg'))
            print(f'{filename} != {text}')


def run(device: str, save_image: bool):
    for idx, screenshot in enumerate(iio.imiter(device)):
        if save_image:
            iio.imsave(f'image_{idx}', screenshot)

        img = process_image(screenshot)

        text = process_text(pytesseract.image_to_string(img, lang='eng'))

        try:
            gTTS(text=text, lang='en').save('output.mp3')

            for _ in range(3):
                playsound('output.mp3')
                time.sleep(1)

            time.sleep(2)
        except AssertionError:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'A OCR-text-speech conversion app')
    parser.add_argument('-t', '--test', help='Test performance on dataset', action='store_true')
    parser.add_argument('-d', '--device', help='/dev/video<N>', default='0')
    parser.add_argument('-s', '--save-image', help='/dev/video<N>', action='store_true')
    args = parser.parse_args()

    if args.test:
        test(args.save_image)
    else:
        run(f'<video{args.device}>', args.save_image)
