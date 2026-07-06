def extract_patch(img_array):
    img_array = np.clip(img_array, -1000, 400)
    return (img_array + 1000) / 1400
