import os
import json

import responder
from text_vectorian import SentencePieceVectorian
from text_vectorian import Char2VecVectorian
import numpy as np


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
MODEL = env['MODEL']

api = responder.API(debug=DEBUG)
vectorian = SentencePieceVectorian() if MODEL == 'sentence-piece' else Char2VecVectorian()


def get_emb(text):
    vectors = vectorian.fit(text).vectors
    return np.mean(vectors, axis=0).tolist()


@api.route("/")
async def encode(req, resp):
    body = await req.text
    texts = json.loads(body)
    emb_list = [get_emb(text) for text in texts]
    if emb_list:
        resp_dict = dict(data=emb_list, dim=len(emb_list[0]))
        resp.media = resp_dict
    else:
        resp.media = dict(data=list())


if __name__ == "__main__":
    api.run()