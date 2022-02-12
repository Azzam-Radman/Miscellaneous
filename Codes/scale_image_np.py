# feature map shape = (3, 3, 128) -----> desired shape (28, 28, 128)
features_for_img_scaled = sp.ndimage.zoom(features_for_img, (28/3, 28/3, 1), order=2)
