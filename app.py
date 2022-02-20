import streamlit as st
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import math
import json
from urllib.request import Request, urlopen
from urllib.parse import quote_plus
from urllib.request import urlopen
from multiprocessing.pool import ThreadPool
from PIL import Image
import clip
import torch
import numpy as np
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
from io import BytesIO

unsplash_access_key = 'CD3pcpVIaLzJ9_EZKFTZBOFW-gEzpdXmpGZ6kyp5szc'

st.title('Imagery Generator')

description = st.text_input('Enter Some Text')
num = st.slider("Number of Pictures", min_value = 1, max_value = 15)
show = st.button("Show Me Some Pictures!")

search_count = 100

# Function used to load a photo from the API
# The photos are downloaded in a small resolution (max 500 pixels wide), because CLIP only supports 224x224 images
def load_photo(url):
    return Image.open(urlopen(url + "&w=500"))  

def get_image_download_link(img, filename):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<p style = "color: #9F5FE0;"> <a href="data:file/jpg;base64,{img_str}" download = {filename}> download </a> </p>'
	return href

if show:
    if not description:
        st.write("Please Fill In All Fields")
    else:
        # Convert the search keywords in a format suitable for the API
        query_string = quote_plus(description)

        # Compute how much pages we need to fetch fromt he search results (assuming 20 photos per page)
        photos_per_page = 20
        pages_count = math.ceil(search_count/photos_per_page)

        # Go through each search result page and store the URLs and metadata of the photos
        photos_data = []
        for page in range(0, pages_count):
            # Make an authenticated call to the API and parse the results as JSON
            request = Request(f"https://api.unsplash.com/search/photos?page={page+1}&per_page={photos_per_page}&query={query_string}")
            request.add_header("Authorization", f"Client-ID {unsplash_access_key}")
            response = urlopen(request).read().decode("utf-8")
            search_result = json.loads(response)

            # Add each photo URL to the list
            for photo in search_result['results']:
                photos_data.append(dict(url=photo['urls']['raw'], 
                                    link=photo['links']['html'],
                                    user_name=photo['user']['name'],
                                    user_link=photo['user']['links']['html'],))

        # Parallelize the download using a thread pool
        photo_urls = [photo['url'] for photo in photos_data]
        pool = ThreadPool(16)
        photos = pool.map(load_photo, photo_urls)

        # Load the open CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        with torch.no_grad():
            # Encode and normalize the description using CLIP
            description_encoded = model.encode_text(clip.tokenize(description).to(device))
            description_encoded /= description_encoded.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            # Preprocess all photos and stack them in a batch
            photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

            # Encode and normalize the photos using CLIP
            photos_encoded = model.encode_image(photos_preprocessed)
            photos_encoded /= photos_encoded.norm(dim=-1, keepdim=True)

        # Retrieve the description vector and the photo vectors
        description_vector = description_encoded.cpu().numpy()
        photo_vectors = photos_encoded.cpu().numpy()

        # Compute the similarity between the descrption and each photo using the Cosine similarity
        similarities = list((description_vector @ photo_vectors.T).squeeze(0))

        # Sort the photos by their similarity score
        best_photos = sorted(zip(similarities, range(len(photos))), key=lambda x: x[0], reverse=True)

        from IPython.core.display import HTML

        pil_imgs = []
        for i in range (num):
            pil_imgs.append(photos[best_photos[i][1]])

        block = []
        expected_height = 600
        for img in pil_imgs:
            w, h = img.size
            frac = h / expected_height

            img = img.resize((int(w / frac), expected_height))
            img = np.array(img)
            block.append(img)

        mh = sum([x.shape[0] for x in block])
        mw = sum([x.shape[1] for x in block])

        grid = np.zeros(shape=(expected_height, mw, 3), dtype=np.uint8)

        running_w = 0
        for img in block:
            h, w = img.shape[:2]
            grid[:, running_w : running_w + w, :] = img
            running_w += w

        st.image(grid, use_column_width= True)
        gridimg = Image.fromarray(grid) 
        col1, col2 = st.columns( [0.9, 0.1])
        with col2:
            st.markdown(get_image_download_link(gridimg, "images.png"), unsafe_allow_html=True)
        credits = []
        with st.expander("Image Credits"):
            for i in range(num):
                photo_data = photos_data[best_photos[i][1]]
                st.markdown("Photo " + str(i + 1) + " By " + photo_data["user_name"] + " On Unsplash: " + photo_data["link"], unsafe_allow_html=True)

        

        
