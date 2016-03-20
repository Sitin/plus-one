from twitter import *
from secrets import *


def make_post(status="", image_path=None):
    status += " #ermilovcentre"
    
    twitter = Twitter(auth=OAuth(token, token_key, api_secret, api_secret_key))

    if image_path is not None:
        with open(image_path, "rb") as imagefile:
            imagedata = imagefile.read()

        t_up = Twitter(domain='upload.twitter.com',
            auth=OAuth(token, token_key, api_secret, api_secret_key))

        post_img_id = t_up.media.upload(media=imagedata)["media_id_string"]

        # - finally send your tweet with the list of media ids:
        twitter.statuses.update(status=status, media_ids=post_img_id)
    else:
        twitter.statuses.update(status=status)