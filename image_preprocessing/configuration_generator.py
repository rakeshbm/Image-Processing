import json

parameters = {
    # guassian
    'guassian': [{
        'gamma_width': 3,
        'gamma_height': 3,
        'gamma_blur': 5
    }, {
        'gamma_width': 3,
        'gamma_height': 3,
        'gamma_blur': 5
    }],

    # rescale
    'scale': [{
        'x': 2,
        'y': 2,
    }, {
        'x': 0.5,
        'y': 5,
    }],

    # sharpening
    'sharpening': [{
        'first_filter': 3,
        'second_filter': 1,
        'alpha': 30
    }],

    # median blur
    'median_blur': [{
        'median_blur_strength': 5
    }],

    # bilateral blur
    'bilateral_blur': [{
        'diameter': 9,
        'sigma_color': 75,
        'sigma_space': 75
    }],

    # denoising
    'denoising': [{
        'filter_length': 5,
        'color_component': 10,
        'temp_window_size': 7,
        'search_window_size': 21,
    }, {
        'filter_length': 10,
        'color_component': 10,
        'temp_window_size': 7,
        'search_window_size': 21,
    }],

    # brightness
    'bright': [{
        "delta": 30
    }],

    # contrast
    'contrast': [{
        "delta": 1.5
    }],
    'perspective_transform': [{
        "p": 0.3,
        "flag": 1
    }, {
        "p": 0.3,
        "flag": 2
    }, {
        "p": 0.3,
        "flag": 3
    }, {
        "p": 0.3,
        "flag": 4
    }]

    # hue
    # 'hue': 0.1,

    # gamma
    # 'gamma': 2,

    # saturation
    # 'saturation': 0.75,

}

with open('./config.json', 'w') as json_file:
    json.dump(parameters, json_file)


