import flask
import requests
import timeit
from cStringIO import StringIO

import predictor


app = flask.Flask(__name__)


@app.route('/api/breed/', methods=['GET'])
def predict_breed_from_image_url():
    image_url = flask.request.args.get('url')
    if not image_url:
        return flask.jsonify({
            'error': 'Need to specify an image url as ?url=',
        })

    start_time = timeit.default_timer()
    image_url = flask.request.args['url']
    image_file = fetch_remote_file(image_url)
    fetch_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    breed = predictor.predict_breed(image_file)
    prediction_time = timeit.default_timer() - start_time

    return flask.jsonify({
        'breed': breed,
        'fetch_time': fetch_time,
        'prediction_time': prediction_time,
    })


def fetch_remote_file(url):
    response = requests.get(url)
    f = StringIO()
    f.write(response.content)
    f.seek(0)
    return f

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
