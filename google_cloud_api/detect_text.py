# https://cloud.google.com/vision/docs/detect-labels-image-client-libraries

import glob
from google.cloud import vision
import io
import re
import pathlib


client = vision.ImageAnnotatorClient()

def detect_document(path):
    """Detects document features in an image."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image
        # , image_context={'language_hints': ['ja']}
        )
    word_texts = ''
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            # print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                # print('Paragraph confidence: {}'.format(
                #     paragraph.confidence))
                # print(f'Paragraph.words {paragraph.words}')
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    # print('Word text: {} (confidence: {})'.format(
                    #     word_text, word.confidence))

                    word_texts += word_text

                    # for symbol in word.symbols:
                    #     print('\tSymbol: {} (confidence: {})'.format(
                    #         symbol.text, symbol.confidence))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return word_texts

def extruct_zip(text):
    zip_pattern = '.*(\d{3}-\d{4}).*'
    matched = re.match(zip_pattern, text)
    if matched:
        zip = matched.group(1)
        zip = zip.replace('-','')
        return zip
    else:
        zip_pattern = '.*(\d{7}).*'
        matched = re.match(zip_pattern, text)
        if matched:
            zip = matched.group(1)
            return zip
        else:
            return 'zip not found ...'

def invoke_zip_checker():
    paths = glob.glob('../data/zip/*.png')
    for path in paths:
        filename = pathlib.Path(path).name
        print(f'\n\n\n-- File name --: {filename}')
        text = detect_document(path)
        print(text)
        extruct_zip(text)
        print(f'Extruct zip: {zip}')
        print(f'Actual zip:  {path[-11:-4]}')

def invoke_message_checker():
    paths = glob.glob('../data/message/*.jpg')
    for path in paths:
        filename = pathlib.Path(path).name
        print(f'\n\n\n-- File name --: {filename}')
        text = detect_document(path)
        print(text)

# invoke_zip_checker()
invoke_message_checker()