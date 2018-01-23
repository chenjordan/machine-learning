from flask import Flask, request, render_template

from image_classify.classify_api import classify
from PIL import Image

import flask
import werkzeug
import os
import datetime
import exifutil
import cStringIO as StringIO
import urllib2

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpe', 'jpeg'])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def cls():
    if request.files.get('file'):
        file = request.files.get('file')
        data = file.read()
        results = classify(data)
        # return jsonify({'res': classify(data)})
        return render_template('result.html', **locals())
    else:
        return index()

@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # process upload image
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        
        path, extension = os.path.splitext(filename)

        if extension == '.png':
            im = Image.open(filename)
            filename = "%s.jpg" % path
            im.save(filename)
        print('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        print('Uploaded image open error: %s', err)
        return flask.render_template(
                'index.html', has_result=True,
                result=(False, 'Cannot open uploaded image.')
            )
    data = open(os.path.join(filename), 'rb').read()
    results = classify(data)
    #print('classify result %s' % results)
    
    return flask.render_template(
            'index.html', has_result=True, result=[True, parse_classify_result(results)],
            imagesrc=embed_image_html(image))

    """
    names, probs, time_cost, accuracy = app.clf.classify_image(
            open(os.path.join(filename), "rb").read())
    
    retrun flask.render_template(
        'index.html', has_result=True, result=[True, zip(names, probs), '%.3f' % time_cost],
        imagesrc=embed_image_html(image)
    )
   """ 

def parse_classify_result(result_ori):
    result_list = []

    for i in result_ori:
        if float(i['score']) >= 0.05:
            result_list.append((i['label'], i['score']))
    print('result_list %s' % result_list)
    return result_list


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        bytes = urllib2.urlopen(imageurl).read()
        string_buffer = StringIO.StringIO(bytes)
        image = exifutil.open_oriented_im(string_buffer)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just not continue
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )
    
    app.logger.info('Image: %s', imageurl)

    results = classify(bytes)
    #print('classify result %s' % results)

    return flask.render_template(
        'index.html', has_result=True, result=[True, parse_classify_result(results)],
        imagesrc=embed_image_html(image))





if __name__ == '__main__':
    app.run(debug=True, processes=1, host='0.0.0.0', port=7100)
