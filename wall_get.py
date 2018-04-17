# coding: utf-8

import vk_api
import json
from constants import *
import requests


def get_post_comments(comment):
    '''
    comments, answered to somebody ('reply_to_user'), 
    generally don't keep information about the post.
    '''
    
    comments_list = []
    for item in comment['items']:
        if (not 'reply_to_user' in item):
            comments_list.append(item['text'])
        
    return comments_list


def get_post_text(post):
    '''
    the post can be resent from other wall
    '''
    
    if 'copy_history' in post:
        text = post['copy_history'][0]['text']
    else:
        text = post['text']
    return text


music = {}
posts = []

# now I have not limited in time token
vk_session = vk_api.VkApi(app_id = app_id, token = token)
vk_session.auth()

tools = vk_api.VkTools(vk_session)

# generator
wall = tools.get_all_iter('wall.get', 1, {'owner_id': owner_id})
step = 0

while step < step_num + 1:
    # for some reasons the first and the second posts are
    # the same (have the same id), you can't see it via brouser
    try:
        curr_post = next(wall)
    except StopIteration:
        print("end of generator")
        print(step)
        break
    curr_post_id = curr_post['id']
    comment = tools.get_all(
        'wall.getComments', 1, {'owner_id': owner_id,
                                'post_id': curr_post_id})
    if comment['count'] == 0: continue
    music[curr_post_id] = {'comments': [], 'text': []}

    # get post description (text)
    music[curr_post_id]['text'] = get_post_text(curr_post)
    
    # get comments
    music[curr_post_id]['comments'] = get_post_comments(comment)
    step += 1

