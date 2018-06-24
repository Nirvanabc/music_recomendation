# coding: utf-8

import vk_api
import json
from constants import *
import requests


def filter_bad_characters(text):
    
    return text  # .replace('\n', '')


def get_post_comments(comment):
    '''
    comments, answered to somebody ('reply_to_user'), 
    generally don't keep information about the post,
    so we omit them.
    '''
    
    comments_list = []
    for item in comment['items']:
        if (not 'reply_to_user' in item):
            comments_list.append((
                filter_bad_characters(item['text']),
                filter_bad_characters(item['from_id'])))
    return comments_list


def get_post_text(post):
    '''
    the post can be resent from other wall (copy_history)
    so there are two variants
    '''
    
    if 'copy_history' in post:
        text = post['copy_history'][0]['text']
    else:
        text = post['text']
    return filter_bad_characters(text)


music = {}
commentators = {}

# now I am not limited in time token
vk_session = vk_api.VkApi(app_id = app_id, token = token)
vk_session.auth()

tools = vk_api.VkTools(vk_session)

# generator
wall = tools.get_all_iter('wall.get', 1, {'owner_id': owner_id})
step = 0

## TEST TRY_EXCEPT block, it may work bad!

while step < step_num + 1:
    # for some reasons the first and the second posts are
    # the same (have the same id), you can't see it via brouser,
    # so step_num + 1
    try:
        curr_post = next(wall)
    except StopIteration:
        print("end of generator")
        print(step)
        break
    except ConnectionError:
        print("ConnectionError, don't warry")
        continue
    curr_post_id = curr_post['id']
    comment = tools.get_all(
        'wall.getComments', 1, {'owner_id': owner_id,
                                'post_id': curr_post_id})

    # if there is no comments
    if comment['count'] == 0: continue

    # music = {'post_id':'comments':[(id, text), ..], 'text': ''}
    
    music[curr_post_id] = {'comments': [], 'text': []}

    # get post description (text)
    music[curr_post_id]['text'] = get_post_text(curr_post)
    
    # get comments
    music[curr_post_id]['comments'] = get_post_comments(comment)
    print(step)
    step += 1


## store data

with open('data.json', 'w') as outfile:
    json.dump(music, outfile)
   
infile = open('data.json')
# 
# # load data
# data_from_file = json.load(infile)
